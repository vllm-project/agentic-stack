"""
Adapted from
https://github.com/gdavid7/cryptocode/blob/main/cryptocode.py
but with simpler code and fixed empty salt to enable DB lookup.
"""

import base64
import hashlib
import secrets
from base64 import b64decode, b64encode
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from hashlib import blake2b
from typing import Any

from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.x509.oid import NameOID


def _encrypt(message: str, password: str, aes_mode: int) -> str:
    """
    Encrypts a message using AES encryption with the given password and mode.
    :param message: The message to encrypt.
    :param password: The password to use for encryption.
    :param aes_mode: The AES mode to use (either AES.MODE_SIV or AES.MODE_GCM).
    :return: The encrypted message as a string.
    """
    if not (aes_mode == AES.MODE_SIV or aes_mode == AES.MODE_GCM):
        raise ValueError("`aes_mode` can only be `AES.MODE_SIV` or `AES.MODE_GCM`.")

    # Use the Scrypt KDF to get a private key from the password
    if aes_mode == AES.MODE_SIV:
        salt = b""
    elif aes_mode == AES.MODE_GCM:
        salt = get_random_bytes(AES.block_size)
    else:
        raise ValueError("`aes_mode` can only be `AES.MODE_SIV` or `AES.MODE_GCM`.")
    private_key = hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    # Create cipher config
    cipher_config = AES.new(private_key, aes_mode)
    # Encrypt the message
    cipher_text, tag = cipher_config.encrypt_and_digest(message.encode("utf-8"))
    # Encode the cipher_text and tag to base64
    cipher_text_b64 = b64encode(cipher_text).decode("utf-8")
    tag_b64 = b64encode(tag).decode("utf-8")

    # Create final encrypted text
    if aes_mode == AES.MODE_SIV:
        encrypted = f"{cipher_text_b64}*{tag_b64}"
    else:
        salt_b64 = b64encode(salt).decode("utf-8")
        nonce_b64 = b64encode(cipher_config.nonce).decode("utf-8")
        encrypted = f"{cipher_text_b64}*{salt_b64}*{nonce_b64}*{tag_b64}"

    return encrypted


def encrypt_random(message: str, password: str) -> str:
    """
    Encryption using AES GCM mode with random salt and nonce.
    """
    return _encrypt(message, password, AES.MODE_GCM)


@lru_cache(maxsize=100000)
def encrypt_deterministic(message: str, password: str) -> str:
    """
    Deterministic encryption using AES SIV mode with
    fixed empty salt and without nonce to enable DB lookup.
    """
    return _encrypt(message, password, AES.MODE_SIV)


@lru_cache(maxsize=100000)
def decrypt(encrypted: str, password: str) -> str:
    """
    Decrypts an encrypted message using AES decryption with the given password.

    :param encrypted: The encrypted message as a string.
    :param password: The password used for decryption.
    :return: The decrypted message as a string.
    """
    parts = encrypted.split("*")
    n_parts = len(parts)

    # Decode the entries from base64
    if n_parts == 4:
        cipher_text_b64, salt_b64, nonce_b64, tag_b64 = parts
        salt = b64decode(salt_b64)  # Decode salt to bytes
        nonce = b64decode(nonce_b64)  # Decode nonce to bytes
    elif n_parts == 2:
        cipher_text_b64, tag_b64 = parts
        salt = b""  # Use empty salt for AES.MODE_SIV
        nonce = None  # No nonce for AES.MODE_SIV
    else:
        raise ValueError(f"Encrypted string must have either 2 or 4 parts, received: {n_parts}")

    # Decode cipher_text and tag to bytes
    cipher_text = b64decode(cipher_text_b64)
    tag = b64decode(tag_b64)

    # Generate the private key from the password and salt
    private_key = hashlib.scrypt(
        password.encode(),  # Encode password to bytes
        salt=salt,  # salt is already bytes
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )

    # Create the cipher config
    cipher: Any  # Use Any to avoid issues with inaccessible types
    if n_parts == 4:
        cipher = AES.new(private_key, AES.MODE_GCM, nonce=nonce)  # Use GCM mode with nonce
    else:
        cipher = AES.new(private_key, AES.MODE_SIV)  # Use SIV mode

    # Decrypt the cipher text
    decrypted = cipher.decrypt_and_verify(cipher_text, tag)  # Both inputs are bytes
    return decrypted.decode("UTF-8")  # Decode the decrypted bytes to a string


def hash_string_blake2b(string: str, key_length: int = 8) -> str:
    if key_length % 2 != 0:
        raise ValueError("Key length must be a multiple of 2.")
    hasher = blake2b(digest_size=key_length // 2)  # 2 characters per byte
    hasher.update(string.encode())
    return hasher.hexdigest()


def hash_file(file, hasher_constructor, ashexstr: bool = False, blocksize: int = 2**10):
    hasher = hasher_constructor()
    block = file.read(blocksize)
    while len(block) > 0:
        hasher.update(block)
        block = file.read(blocksize)
    return hasher.hexdigest() if ashexstr else hasher.digest()


def blake2b_hash_file(file, blocksize: int = 2**10):
    return hash_file(file, hasher_constructor=blake2b, ashexstr=True, blocksize=blocksize)


def generate_key(key_length: int = 48, prefix: str = "") -> str:
    """
    Generates a random key.

    Args:
        key_length (int, optional): The desired length of the key. Defaults to 48.
        prefix (str, optional): Prefix of the key. Defaults to "".

    Raises:
        ValueError: If `key_length` is < 16 or not a multiple of 2.

    Returns:
        api_key (str): A random key.
    """
    if key_length < 16:
        raise ValueError("Key length must be at least 16 characters.")
    if key_length % 2 != 0:
        raise ValueError("Key length must be a multiple of 2.")
    api_key = blake2b(secrets.token_bytes(key_length), digest_size=key_length // 2).hexdigest()
    return f"{prefix}{api_key}"


def b64url_uint(n: int) -> str:
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def b64_der_no_wrap(der_bytes: bytes) -> str:
    # For x5c, use standard base64 (not URL-safe) and no padding stripping.
    # Spec allows normal base64; common practice is standard base64 without newlines.
    return base64.b64encode(der_bytes).decode("ascii")


def make_self_signed_cert(private_key, subject_cn: str, days_valid: int = 365):
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Org"),
            x509.NameAttribute(NameOID.COMMON_NAME, subject_cn),
        ]
    )
    now = datetime.now(timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=days_valid))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(private_key.public_key()), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(private_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [x509.ExtendedKeyUsageOID.SERVER_AUTH, x509.ExtendedKeyUsageOID.CLIENT_AUTH]
            ),
            critical=False,
        )
    )
    cert = builder.sign(private_key=private_key, algorithm=hashes.SHA256())
    return cert


def public_key_to_jwk(
    private_pem: bytes,
    public_pem: bytes,
    kid: str = "6IHV-hh-U-gue_Z5xBgKi",
):
    private_key = serialization.load_pem_private_key(private_pem, password=None)
    public_key = serialization.load_pem_public_key(public_pem)
    numbers = public_key.public_numbers()
    # Create self-signed certificate
    cert = make_self_signed_cert(private_key, subject_cn="test.example")
    der = cert.public_bytes(serialization.Encoding.DER)
    # x5c is an array of base64-encoded (not URL-safe) DER certs, leaf first
    x5c = [b64_der_no_wrap(der)]
    # x5t (SHA-1) and x5t#S256 (SHA-256) of the DER
    x5t = base64.urlsafe_b64encode(hashlib.sha1(der).digest()).rstrip(b"=").decode("ascii")
    jwk = {
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": kid,
        "n": b64url_uint(numbers.n),
        "e": b64url_uint(numbers.e),
        "x5c": x5c,
        "x5t": x5t,
    }
    return jwk


def public_key_from_x5c(jwk: dict[str, str | list[str]]):
    x5c_list = jwk.get("x5c")
    if not x5c_list:
        raise ValueError("No x5c chain in JWK")
    cert_der = base64.b64decode(x5c_list[0])  # first element is leaf cert
    cert = x509.load_der_x509_certificate(cert_der, default_backend())
    return cert.public_key()
