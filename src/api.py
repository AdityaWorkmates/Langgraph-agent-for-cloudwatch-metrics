import os
import logging
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

from src.routes import main_routes
from src.config import LOG_LEVEL

from flasgger import Swagger

load_dotenv()

app = Flask(__name__)
CORS(app)
Swagger(app)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app.register_blueprint(main_routes)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 6000)), debug=True)