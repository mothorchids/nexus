import os
from dotenv import load_dotenv

import qnexus as qnx
from qnexus.client.auth import login_no_interaction

load_dotenv()
username = os.getenv("HQS_USERNAME")
password = os.getenv("HQS_PASSWORD")

if username is None or password is None:
    raise RuntimeError("HQS_USERNAME or HQS_PASSWORD missing in .env file.")

login_no_interaction(user=username, pwd=password)

devices = qnx.devices.get_all()

print(devices)

qnx.logout()
