

from apis.reports.combined_kde import combined_kde
from apis.reports.player_distances import player_distances
from apis.reports.player_ids import player_ids
from apis.reports.team_distances import team_distances
from apis.reports.player_passings import player_passings
from apis.reports.team_passings import team_passings
from apis.reports.team_timeline import team_timeline
from apis.reports.team_pie_passings import team_pie_passings
from apis.reports.player_pie_passings import player_pie_passings

def register_reports_blueprints(app):
    app.register_blueprint(combined_kde)
    app.register_blueprint(player_distances)
    app.register_blueprint(player_ids)
    app.register_blueprint(team_distances)
    app.register_blueprint(player_passings)
    app.register_blueprint(team_passings)
    app.register_blueprint(team_timeline)
    app.register_blueprint(team_pie_passings)
    app.register_blueprint(player_pie_passings)
