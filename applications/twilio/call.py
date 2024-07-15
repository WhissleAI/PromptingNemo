# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client

# Set environment variables for your credentials
# Read more at http://twil.io/secure

account_sid = "AC036b65392d3ed4c975b62dfdd94dd6d8"
auth_token = "67860a42a80d84e2cd2aa4e36443d616"
client = Client(account_sid, auth_token)

call = client.calls.create(
  url="http://demo.twilio.com/docs/voice.xml",
  to="+12138224814",
  from_="+18336901469"
)

print(call.sid)
