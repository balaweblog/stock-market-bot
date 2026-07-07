import smtplib
from email.mime.text import MIMEText
from config import *
import re


def send_email(body, html=False):
    if html:
        msg = MIMEText(body, "html")
    else:
        msg = MIMEText(body)

    clean_email_to = EMAIL_TO.replace("\n", "").replace("\r", "")

    recipients = [
        email.strip()
        for email in clean_email_to.split(",")
        if email.strip()
    ]
    EMAIL_REGEX = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

    recipients = [e for e in recipients if re.match(EMAIL_REGEX, e)]

    msg["Subject"] = "AI Stock Report"
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)

    print("RAW EMAIL_TO:", repr(EMAIL_TO))
    print("RECIPIENTS:", recipients)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(EMAIL_FROM, EMAIL_PASSWORD)
    server.sendmail(EMAIL_FROM, recipients, msg.as_string())
    server.quit()