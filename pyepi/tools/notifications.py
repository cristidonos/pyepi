import smtplib
from email.mime.text import MIMEText
import pathlib
import os


def sendmail(to_address=None, subject=None, message=None, from_address='Pyepi Bot', smtp=None, username=None,
             password=None, credentials_file='credentials.txt'):
    """Email Sender

    Parameters
    ----------
    to_address: string
        To email address.
    subject: string
        Email subject line
    message: string
        Message body
    from_address: string
        Sender address
    smtp: string
        SMTP server and port ('smtp.gmail.com:587')
    username: string
        Account's username
    password: string
        Account's password
    credentials_file: string
        Filename containing email credentials.
        It's a 3 lines text file living in current user's folder, containing:
            1. the server's SMTP ('smtp.gmail.com:587')
            2. username
            3. password

    """
    if smtp is None or username is None or password is None:
        try:
            dir = str(pathlib.Path.home())
            with open(os.path.join(dir, credentials_file)) as c:
                lines = c.readlines()
                smtp = lines[0].strip().split(sep=':')
                username = lines[1].strip()
                password = lines[2].strip()
                print('done reading file')
        except:
            # unspecified SMTP and credentials
            print('\n\n')
            print(''' --> Email notification not sent, likely due to missing credentials file.''')
            return 0

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = from_address
    msg['To'] = to_address

    server = smtplib.SMTP(host=smtp[0], port=int(smtp[1]))
    server.starttls()
    server.login(username, password)
    server.send_message(msg)
    server.quit()
