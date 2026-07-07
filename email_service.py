import smtplib
from email.mime.text import MIMEText
from config import *


def send_email(body, html=False):
    if html:
        msg = MIMEText(body, "html")
    else:
        msg = MIMEText(body)

    recipients = parse_email_list(EMAIL_TO)
    cc_recipients = parse_email_list(EMAIL_CC)

    if not recipients:
        print("No valid TO recipients found. Please set EMAIL_TO.")
        return

    msg["Subject"] = "AI Stock Report"
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)
    if cc_recipients:
        msg["Cc"] = ", ".join(cc_recipients)

    all_recipients = recipients + cc_recipients

    print("RAW EMAIL_TO:", repr(EMAIL_TO))
    print("RAW EMAIL_CC:", repr(EMAIL_CC))
    print("TO RECIPIENTS:", recipients)
    print("CC RECIPIENTS:", cc_recipients)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(EMAIL_FROM, EMAIL_PASSWORD)
    server.sendmail(EMAIL_FROM, all_recipients, msg.as_string())
    server.quit()