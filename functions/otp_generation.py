# import secrets

# def generate_secure_otp(length=6):
#     digits = "0123456789"
#     otp = "".join(secrets.choice(digits) for _ in range(length))
#     return otp

# print("Secure OTP:", generate_secure_otp())

import string
import secrets

def generate_alphanumeric_otp(length=6):
    chars = string.ascii_letters + string.digits
    otp = ''.join(secrets.choice(chars) for _ in range(length))
    return otp

print("OTP:", generate_alphanumeric_otp())