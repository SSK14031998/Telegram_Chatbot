from telegram import update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

with open('token.txt','r') as f: #open the token file for the bot
    TOKEN=f.read().strip()

x_train, y_train, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Frog', 'Horse']

# Neural Network
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,(3,3), activation='relu'),
    tf.keras.layers.Dense(64,(3,3), activation='softmax'),
])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome!")
    
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""
    /start - Starts the conversation
    /help -  Shows this message
    /train - Trains Neural Network
    """)

async def train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Model is being trained...")
    logging.info("Start model training....")

    try:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train,y_trian, epochs=10, validation_data=(x_test, y_test))
        model.save('cifar_classifier.model')
        logging.info("Model training completed and saved.")
        await update.message.reply_text("Done! You can now send a photo")
    except Exception as e:
        logger.error(f"An Error Occured during training: (e)")
        await update.message.reply_text(f"An Error Occured during training: (e)")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Train the model and send a picture")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await context.bot.get_file(update.message.photo[-1].file_id)

    file_bytes = np.asarray(bytearray(f.read()), dtype=np.unit8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)

    prediction = model.prediction(np.array([img / 255.0]))
    await update.message.reply_text(f"In this image I see a {class_names[np.argmax(prediction)]}")

app = Application.builder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("help", help_command))
app.add_handler(CommandHandler("train", train))

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

app.run_polling()
