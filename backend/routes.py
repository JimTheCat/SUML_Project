# routes.py

from flask import Blueprint, jsonify, request

from models import Post, posts

api_blueprint = Blueprint('api', __name__)


# Endpoint do pobierania wszystkich postów
@api_blueprint.route('/posts', methods=['GET'])
def get_posts():
    return jsonify([post.to_dict() for post in posts])


# Endpoint do pobierania pojedynczego posta
@api_blueprint.route('/posts/<int:post_id>', methods=['GET'])
def get_post(post_id):
    post = next((post for post in posts if post.id == post_id), None)
    if post is None:
        return jsonify({'error': 'Post not found'}), 404
    return jsonify(post.to_dict())


# Endpoint do tworzenia nowego posta
@api_blueprint.route('/posts', methods=['POST'])
def create_post():
    new_post_data = request.get_json()
    new_post = Post(len(posts) + 1, new_post_data['title'], new_post_data['content'])
    posts.append(new_post)
    return jsonify(new_post.to_dict()), 201


# Endpoint do aktualizacji posta
@api_blueprint.route('/posts/<int:post_id>', methods=['PUT'])
def update_post(post_id):
    post = next((post for post in posts if post.id == post_id), None)
    if post is None:
        return jsonify({'error': 'Post not found'}), 404

    updated_data = request.get_json()
    post.title = updated_data.get('title', post.title)
    post.content = updated_data.get('content', post.content)
    return jsonify(post.to_dict())


# Endpoint do usuwania posta
@api_blueprint.route('/posts/<int:post_id>', methods=['DELETE'])
def delete_post(post_id):
    p = [post for post in posts if post.id != post_id]
    return jsonify({'result': 'Post deleted'})
