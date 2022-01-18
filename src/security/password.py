from passlib.context import CryptContext


class PasswordContext:
    def __init__(self):
        self.context = CryptContext(
            schemes=['bcrypt'],
            deprecated='auto',
        )

    def hash(self, password):
        return self.context.hash(password)

    def verify(self, password, password_hash):
        return self.context.verify(password, password_hash)
