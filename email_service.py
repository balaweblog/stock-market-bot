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

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(EMAIL_FROM, EMAIL_PASSWORD)
    server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    server.quit()