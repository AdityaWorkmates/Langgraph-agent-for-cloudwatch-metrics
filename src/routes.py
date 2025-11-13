from flasgger import swag_from
import json
import logging
import time

from flask import request, jsonify, Blueprint
from flask_cors import CORS

from werkzeug.utils import secure_filename
from langgraph.graph import StateGraph, START, END

from src.utils import (
    State,
    parse_input,
    analyze_with_llm,
    generate_plots,
    format_output,
    run_graph
)
from src.config import MODEL_ID

logger = logging.getLogger(__name__)

main_routes = Blueprint('main_routes', __name__)









graph_builder = StateGraph(State)

graph_builder.add_node("parse_input", parse_input)
graph_builder.add_node("analyze_with_llm", analyze_with_llm)
graph_builder.add_node("generate_plots", generate_plots)
graph_builder.add_node("format_output", format_output)

graph_builder.add_edge(START, "parse_input")
graph_builder.add_edge("parse_input", "analyze_with_llm")
graph_builder.add_edge("analyze_with_llm", "generate_plots")
graph_builder.add_edge("generate_plots", "format_output")
graph_builder.add_edge("format_output", END)

graph = graph_builder.compile()







@main_routes.route('/health', methods=['GET'])
def health():
    """Health check endpoint.
    ---
    responses:
      200:
        description: The application is healthy.
    """
    logger.info("Health check requested")
    response = jsonify({"status": "ok", "model_id": MODEL_ID})
    logger.info(f"Health check response: {response.get_data(as_text=True)}")
    return response

@main_routes.route('/analyze', methods=['POST'])
def analyze():
    """Analyzes the given payload.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: false
        description: The JSON file to analyze.
      - name: body
        in: body
        required: false
        schema:
          type: object
    responses:
      200:
        description: The analysis was successful.
      400:
        description: The request was invalid.
      500:
        description: An error occurred during the analysis.
    """
    req_id = getattr(request, "environ", {}).get("REQUEST_ID") or str(int(time.time() * 1000))
    logger.info(f"Analyze request {req_id} received")
    logger.info(f"Request headers: {request.headers}")
    logger.info(f"Request body: {request.get_data(as_text=True)}")
    payload = None
    if "file" in request.files:
        f = request.files["file"]
        filename = secure_filename(f.filename or "upload.json")
        logger.info(f"Processing uploaded file: {filename}")
        try:
            payload = json.load(f)
        except Exception as e:
            logger.warning(f"Invalid JSON file uploaded: {e}")
            return jsonify({"error": "invalid json file", "detail": str(e)}), 400
    else:
        try:
            payload = request.get_json(force=True)
        except Exception as e:
            logger.warning(f"Invalid JSON body: {e}")
            return jsonify({"error": "no file and invalid/empty JSON body", "detail": str(e)}), 400

    if not payload:
        logger.warning("Empty payload received")
        return jsonify({"error": "empty payload"}), 400
    try:
        output = run_graph(graph, payload)
        logger.info(f"Analysis successful for request {req_id}")
        logger.debug(f"Analysis output: {output}")
        return jsonify(output), 200

    except Exception as e:
        logger.exception(f"Graph run failed for request {req_id}: {e}")
        return jsonify({"error": "graph run failed", "detail": str(e)}), 500