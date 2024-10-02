# models.py

class Post:
    def __init__(self, post_id, title, content):
        self.id = post_id
        self.title = title
        self.content = content

    def to_dict(self):
        return {'id': self.id, 'title': self.title, 'content': self.content}


# Przykładowe dane
posts = [
    Post(1, 'Pierwszy post', 'To jest zawartość pierwszego posta.'),
    Post(2, 'Drugi post', 'To jest zawartość drugiego posta.')
]
