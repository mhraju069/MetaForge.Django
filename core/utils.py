from cryptography.fernet import Fernet
from django.conf import settings
fernet = Fernet(settings.ENCRYPTION_KEY)

def encrypt_token(token: str) -> str:
    return fernet.encrypt(token.encode()).decode()

def decrypt_token(token: str) -> str:
    return fernet.decrypt(token.encode()).decode()