"""Tạo self-signed certificate cho Flask HTTPS.

Chạy một lần:
    python web/gen_cert.py

Sinh ra:
    web/cert.pem  — certificate (import vào Windows Trust Store)
    web/key.pem   — private key
"""
import ipaddress
import datetime
from pathlib import Path

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

OUT_DIR = Path(__file__).parent

# Đổi thành IP thực của máy nếu cần
HOSTNAMES = ["localhost", "127.0.0.1", "192.168.1.9"]

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COMMON_NAME, "Flask Dev"),
])

san = []
for h in HOSTNAMES:
    try:
        san.append(x509.IPAddress(ipaddress.ip_address(h)))
    except ValueError:
        san.append(x509.DNSName(h))

cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.datetime.utcnow())
    .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=825))
    .add_extension(x509.SubjectAlternativeName(san), critical=False)
    .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
    .sign(key, hashes.SHA256())
)

cert_path = OUT_DIR / "cert.pem"
key_path  = OUT_DIR / "key.pem"

cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
key_path.write_bytes(key.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.TraditionalOpenSSL,
    serialization.NoEncryption(),
))

print(f"✓ Đã tạo: {cert_path}")
print(f"✓ Đã tạo: {key_path}")
print()
print("=== Bước tiếp theo: Import cert vào Windows ===")
print("1. Mở Run (Win+R) → gõ: certlm.msc")
print("2. Trusted Root Certification Authorities → Certificates")
print("3. Chuột phải → All Tasks → Import")
print(f"4. Chọn file: {cert_path.resolve()}")
print("5. Chọn store: Trusted Root Certification Authorities")
print("6. Finish → Yes")
print()
print("Sau đó chạy server:")
print("   python -m web.app --ssl")
