"""Shared email sending logic for all controllers."""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from utils.config import EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO, EMAIL_CC
from utils.logger import log

def parse_email_list(email_string):
    if not email_string:
        return []
    return [email.strip() for email in email_string.split(',') if email.strip()]

def send_email(subject, html_body, pdf_attachment=None, pdf_filename="report.pdf"):
    """
    Sends an HTML email with an optional PDF attachment.
    Returns True if successful, False otherwise.
    """
    if not all([EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO]):
        log.error(
            "Email credentials not found. "
            "Please set EMAIL_FROM, EMAIL_PASSWORD, and EMAIL_TO environment variables."
        )
        return False

    to_recipients = parse_email_list(EMAIL_TO)
    cc_recipients = parse_email_list(EMAIL_CC)

    if not to_recipients:
        log.error("No valid TO recipients found. Please set EMAIL_TO with a comma-separated list of emails.")
        return False

    if pdf_attachment:
        msg = MIMEMultipart("mixed")
        msg.attach(MIMEText(html_body, "html"))
        attachment_part = MIMEApplication(pdf_attachment, _subtype="pdf", Name=pdf_filename)
        attachment_part["Content-Disposition"] = f'attachment; filename="{pdf_filename}"'
        msg.attach(attachment_part)
    else:
        msg = MIMEText(html_body, "html")

    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(to_recipients)
    if cc_recipients:
        msg["Cc"] = ", ".join(cc_recipients)

    all_recipients = to_recipients + cc_recipients

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=30)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, all_recipients, msg.as_string())
        server.quit()
        return True
    except smtplib.SMTPAuthenticationError:
        log.error(
            "SMTP Authentication Error: The username or password you entered is not correct. "
            "Please check your credentials and App Password if using Gmail."
        )
        return False
    except Exception as e:
        log.error(f"An error occurred while sending the email: {e}", exc_info=True)
        return False
