import requests
def webhook_response(webhook_url, status, code, message, data=None):
    def send(webhook_url, status, code, message, data=None):
        response_data = {
            "status": status,
            "code": code,
            "message": message,
            "data": data,
        }
        if webhook_url and "http" in webhook_url:
            requests.post(webhook_url, json=response_data)
    send(webhook_url, status, code, message, data)
    return None