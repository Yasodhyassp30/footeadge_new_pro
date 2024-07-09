from apis.player_profile.add_player import player_profile


def register_players_blueprints(app):
    app.register_blueprint(player_profile)
