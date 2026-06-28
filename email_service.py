import smtplib
from email.mime.text import MIMEText
from config import *

def send_email(body, html=False):
    if html:
        msg = MIMEText(body, "html")
    else:
        msg = MIMEText(body)
    msg["Subject"] = "AI Stock Report"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    recipients = [
        email.strip()
        for email in EMAIL_TO.split(",")
        if email.strip()
    ]


    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(EMAIL_FROM, EMAIL_PASSWORD)
    server.sendmail(EMAIL_FROM, recipients, msg.as_string())
    server.quit()