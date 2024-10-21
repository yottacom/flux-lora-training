import requests
def webhook_response(training_webhook_url, status, code, message, data=None):
    def send(training_webhook_url, status, code, message, data=None):
        response_data = {
            "status": status,
            "code": code,
            "message": message,
            "data": data,
        }
        if training_webhook_url and "http" in training_webhook_url:
            requests.post(training_webhook_url, json=response_data)
    send(training_webhook_url, status, code, message, data)
    return None