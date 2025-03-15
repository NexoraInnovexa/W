import requests

class PaystackAPI:
    def __init__(self, secret_key):
        self.base_url = "https://api.paystack.co"
        self.secret_key = secret_key

    def initiate_payment(self, amount, email, storyteller_subaccount, platform_subaccount, storyteller_share=0.8):
        url = f"{self.base_url}/transaction/initialize"
        headers = {
            "Authorization": f"Bearer {self.secret_key}",
            "Content-Type": "application/json"
        }
        data = {
            "email": email,
            "amount": int(amount * 100),  # Convert to kobo
            "subaccount": storyteller_subaccount,  # The storyteller's subaccount
            "bearer": "account",  # Indicates the fee is split between accounts
            "split": {
                "subaccounts": [
                    {
                        "subaccount": storyteller_subaccount,
                        "share": storyteller_share  # 80% to storyteller
                    },
                    {
                        "subaccount": platform_subaccount,
                        "share": 1 - storyteller_share  # 20% to platform owner
                    }
                ]
            }
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()
