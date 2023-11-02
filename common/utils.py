from flask import make_response, jsonify, Response

def send_response(data: dict, code: int) -> Response:
    """
    Function to send api response

    Args:
        data (dict): data to be sent
        code (int): status code
    
    Returns:
        Response: flask response
    """
    response = make_response(
        jsonify(
            data
        ),
        code,
    )
    response.headers["Content-Type"] = "application/json"
    return response
