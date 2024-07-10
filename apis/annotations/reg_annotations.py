from apis.annotations.annotations import annotation
from apis.annotations.event_detection import event_detection


def register_annotations_blueprints(app):
    app.register_blueprint(annotation)
    app.register_blueprint(event_detection)
