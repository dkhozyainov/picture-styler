import logging
from multiprocessing import Process
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import filters, MessageHandler, CommandHandler, CallbackQueryHandler, ApplicationBuilder, ContextTypes
from io import BytesIO
from PIL import Image
from requests import post
import json

from styletransfer.styletransfer import StyleTransferType, StyleTransfer, StyleTransferInference, StyleTransferConfig


class InferenceWorker:

    @staticmethod
    def get_worker(token: str, chat_id: int, style_transfer: StyleTransfer,
                   content_image: Image, style_image: Image, config: StyleTransferConfig = None):

        inference = style_transfer.get_inference(content_image, style_image, config)
        worker = InferenceWorker(token, chat_id, inference)
        return worker

    def __init__(self, token: str, chat_id: int, inference: StyleTransferInference):
        self._token = token
        self._chat_id = chat_id
        self._inference = inference

    def _send_result_message(self, result_image):
        bio = BytesIO()
        result_image.save(bio, 'JPEG')
        bio.seek(0)

        # Create keyboard with algorithm selection buttons
        keyboard = [
            [
                InlineKeyboardButton("MSGNet", callback_data="algo_msgnet"),
                InlineKeyboardButton("Magenta", callback_data="algo_magenta")
            ],
            [
                InlineKeyboardButton("Gatys", callback_data="algo_gatys"),
                InlineKeyboardButton("MSGNetCustom", callback_data="algo_msgnet_custom")
            ],
            [InlineKeyboardButton("Current", callback_data="algo_current")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Convert keyboard to JSON string for API call
        keyboard_json = json.dumps(reply_markup.to_dict())

        api_url = f'https://api.telegram.org/bot{self._token}/sendPhoto'
        files = {"photo": bio}
        data = {
            "chat_id": self._chat_id,
            "reply_markup": keyboard_json
        }

        try:
            response = post(api_url, files=files, data=data)
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"Exception: {e}")

    # Defining __call__ method
    def __call__(self):
        result_image = self._inference()
        self._send_result_message(result_image)


class Application:

    def __init__(self, default_style_transfer_type: StyleTransferType = StyleTransferType.MSGNet):

        # style transfer part:
        self._default_style_transfer_type = default_style_transfer_type
        self._user_id_to_first_image = {}  # dict for saving the first photo from a user (we need 2 photos)
        # Cache for style transfer instances: (user_id, algorithm_type) -> StyleTransfer instance
        self._style_transfer_cache = {}

        # telegram bot part:
        with open('../token.txt') as file:
            self._token = file.read()
        self._bot = self._build_and_run_bot()

    def _build_style_transfer(self, style_transfer_type: StyleTransferType):
        if style_transfer_type == StyleTransferType.Gatys:
            from styletransfer.gatys.gatys import Gatys
            return Gatys()

        if style_transfer_type == StyleTransferType.Magenta:
            from styletransfer.magenta.magenta import Magenta
            return Magenta()

        if style_transfer_type == StyleTransferType.MSGNet \
                or style_transfer_type == StyleTransferType.MSGNetCustomTrain:

            from styletransfer.msgnet.msgnet import MSGNet, MSGNetConfig

            model_path = './styletransfer/msgnet/'
            model_path += '21styles.model' if style_transfer_type == StyleTransferType.MSGNet \
                                            else 'my21styles.model'

            config = MSGNetConfig(model_path=model_path)

            return MSGNet(config)

    def _get_style_transfer_for_user(self, user_id: int, style_transfer_type: StyleTransferType) -> StyleTransfer:
        """Get or create a style transfer instance for a user and algorithm type."""
        cache_key = (user_id, style_transfer_type)
        if cache_key not in self._style_transfer_cache:
            self._style_transfer_cache[cache_key] = self._build_style_transfer(style_transfer_type)
        return self._style_transfer_cache[cache_key]

    def _create_algorithm_keyboard(self):
        """Create inline keyboard with algorithm selection buttons."""
        keyboard = [
            [
                InlineKeyboardButton("MSGNet", callback_data="algo_msgnet"),
                InlineKeyboardButton("Magenta", callback_data="algo_magenta")
            ],
            [
                InlineKeyboardButton("Gatys", callback_data="algo_gatys"),
                InlineKeyboardButton("MSGNetCustom", callback_data="algo_msgnet_custom")
            ],
            [InlineKeyboardButton("Current", callback_data="algo_current")]
        ]
        return InlineKeyboardMarkup(keyboard)

    def _build_and_run_bot(self):
        bot = ApplicationBuilder().token(self._token).build()

        # Callback query handler for inline button presses (should be first)
        callback_handler = CallbackQueryHandler(self._on_callback_query)
        bot.add_handler(callback_handler)

        # Command handler for algorithm selection (should be before image handler)
        algorithm_handler = CommandHandler('algorithm', self._on_algorithm_command)
        bot.add_handler(algorithm_handler)

        # Command handler for showing current algorithm
        current_handler = CommandHandler('current', self._on_current_command)
        bot.add_handler(current_handler)

        image_handler = MessageHandler(filters.PHOTO | filters.Document.IMAGE, self._on_image)
        bot.add_handler(image_handler)

        # Should be the last handler:
        all_handler = MessageHandler(filters.ALL, self._on_start)
        bot.add_handler(all_handler)

        # ATTENTION Don't add more handlers here

        bot.run_polling()
        return bot

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        txt = ("Hi! I'm Image Styler. Send me 2 photos. I'll take the content from the first one and the style from "
               "the second one. Then I'll generate a new image with the taken content in the taken style and send it"
               " back.\n\n"
               "Choose an algorithm using the buttons below:")
        keyboard = self._create_algorithm_keyboard()
        await context.bot.send_message(
            chat_id=update.effective_chat.id, 
            text=txt,
            reply_markup=keyboard
        )

    async def _on_algorithm_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /algorithm command to select style transfer algorithm."""
        if not context.args:
            # Show available algorithms
            txt = ("Available algorithms:\n\n"
                   "1. MSGNet - Fastest, but sometimes with visible patterns\n"
                   "2. Magenta - Fast enough\n"
                   "3. Gatys - Nice but slow (may take a few minutes)\n"
                   "4. MSGNetCustomTrain - Custom trained MSGNet\n\n"
                   "Usage: /algorithm <number>\n"
                   "Example: /algorithm 1")
            await context.bot.send_message(chat_id=update.effective_chat.id, text=txt)
            return

        try:
            choice = int(context.args[0])
            algorithm_map = {
                1: StyleTransferType.MSGNet,
                2: StyleTransferType.Magenta,
                3: StyleTransferType.Gatys,
                4: StyleTransferType.MSGNetCustomTrain
            }
            
            if choice not in algorithm_map:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"Invalid choice. Please select a number between 1 and {len(algorithm_map)}."
                )
                return

            selected_type = algorithm_map[choice]
            
            # Store in user_data (user_data is always a dict in python-telegram-bot)
            context.user_data['style_transfer_type'] = selected_type
            
            algorithm_names = {
                StyleTransferType.MSGNet: "MSGNet",
                StyleTransferType.Magenta: "Magenta",
                StyleTransferType.Gatys: "Gatys",
                StyleTransferType.MSGNetCustomTrain: "MSGNetCustomTrain"
            }
            
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Algorithm set to: {algorithm_names[selected_type]}"
            )
            
        except ValueError:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Please provide a valid number."
            )

    async def _on_current_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /current command to show current algorithm."""
        # Get algorithm from user_data or use default
        if context.user_data and 'style_transfer_type' in context.user_data:
            current_type = context.user_data['style_transfer_type']
        else:
            current_type = self._default_style_transfer_type
        
        algorithm_names = {
            StyleTransferType.MSGNet: "MSGNet",
            StyleTransferType.Magenta: "Magenta",
            StyleTransferType.Gatys: "Gatys",
            StyleTransferType.MSGNetCustomTrain: "MSGNetCustomTrain"
        }
        
        keyboard = self._create_algorithm_keyboard()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Current algorithm: {algorithm_names[current_type]}",
            reply_markup=keyboard
        )

    async def _on_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboard buttons."""
        query = update.callback_query
        
        # Answer the callback query to remove loading state
        await query.answer()
        
        callback_data = query.data
        user_id = update.effective_user.id
        chat_id = query.message.chat_id if query.message else update.effective_chat.id
        
        algorithm_map = {
            "algo_msgnet": StyleTransferType.MSGNet,
            "algo_magenta": StyleTransferType.Magenta,
            "algo_gatys": StyleTransferType.Gatys,
            "algo_msgnet_custom": StyleTransferType.MSGNetCustomTrain
        }
        
        algorithm_names = {
            StyleTransferType.MSGNet: "MSGNet",
            StyleTransferType.Magenta: "Magenta",
            StyleTransferType.Gatys: "Gatys",
            StyleTransferType.MSGNetCustomTrain: "MSGNetCustomTrain"
        }
        
        keyboard = self._create_algorithm_keyboard()
        
        if callback_data == "algo_current":
            # Show current algorithm
            if context.user_data and 'style_transfer_type' in context.user_data:
                current_type = context.user_data['style_transfer_type']
            else:
                current_type = self._default_style_transfer_type
            
            # Try to edit message if it has text, otherwise send new message
            try:
                if query.message and query.message.text:
                    await query.edit_message_text(
                        text=f"Current algorithm: {algorithm_names[current_type]}",
                        reply_markup=keyboard
                    )
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"Current algorithm: {algorithm_names[current_type]}",
                        reply_markup=keyboard
                    )
            except Exception:
                # If editing fails, send new message
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"Current algorithm: {algorithm_names[current_type]}",
                    reply_markup=keyboard
                )
        elif callback_data in algorithm_map:
            # Set algorithm
            selected_type = algorithm_map[callback_data]
            context.user_data['style_transfer_type'] = selected_type
            
            # Try to edit message if it has text, otherwise send new message
            try:
                if query.message and query.message.text:
                    await query.edit_message_text(
                        text=f"Algorithm set to: {algorithm_names[selected_type]}",
                        reply_markup=keyboard
                    )
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"Algorithm set to: {algorithm_names[selected_type]}",
                        reply_markup=keyboard
                    )
            except Exception:
                # If editing fails, send new message
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"Algorithm set to: {algorithm_names[selected_type]}",
                    reply_markup=keyboard
                )

    async def _on_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = update.message
        bot = context.bot

        file_id = msg.photo[-1].file_id if msg.photo else msg.document.file_id
        file = await bot.get_file(file_id)
        out = BytesIO()
        await file.download_to_memory(out)
        img = Image.open(out)

        chat_id = msg.chat_id
        user_id = update.effective_user.id
        
        if chat_id not in self._user_id_to_first_image.keys():
            # TODO Doesn't look good if user sends both images at once:
            # txt = "...please send another one for style..."
            # await context.bot.send_message(chat_id=update.effective_chat.id, text=txt)

            self._user_id_to_first_image[chat_id] = img
            print(f"The first image from user {chat_id} has been saved, waiting for the second one")
        else:
            # Get user's selected algorithm or use default
            if context.user_data and 'style_transfer_type' in context.user_data:
                style_transfer_type = context.user_data['style_transfer_type']
            else:
                style_transfer_type = self._default_style_transfer_type
            
            # Get or create style transfer instance for this user and algorithm
            style_transfer = self._get_style_transfer_for_user(user_id, style_transfer_type)
            
            if style_transfer_type == StyleTransferType.Gatys:  # the slow one
                txt = "...in progress. Please wait, it may take a few minutes..."
            else:
                txt = "...in progress. Please wait, it may take for a while..."

            await context.bot.send_message(chat_id=update.effective_chat.id, text=txt)

            print(f"The second image from user {chat_id} has been received and we start the style transfer")
            content_image = self._user_id_to_first_image.pop(chat_id)
            style_image = img

            inference_worker = InferenceWorker.get_worker(self._token, chat_id,
                style_transfer, content_image, style_image)

            p = Process(target=inference_worker)
            p.start()


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # Default style transfer algorithm (can be changed via /algorithm command in Telegram)
    default_style_transfer_type = StyleTransferType.MSGNet

    Application(default_style_transfer_type)
