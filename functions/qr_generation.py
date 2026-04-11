import qrcode

data = "https://www.google.com"

img = qrcode.make(data)
img.save("qrcode.png")

print("QR Code generated!")
# import qrcode

# qr = qrcode.QRCode(
#     version=1,
#     box_size=10,
#     border=5
# )

# qr.add_data("Anmol Gupta - Portfolio")
# qr.make(fit=True)

# img = qr.make_image(fill_color="black", back_color="white")
# img.save("custom_qr.png")