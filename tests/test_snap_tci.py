"""בדיקות ל-_snap_tci_to_latlon_path — הצמדת TCI למסלול חיצוני (OSRM), גרף סינתטי זעיר."""
import networkx as nx

from src.routing import _snap_tci_to_latlon_path


def _tiny_graph():
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    G.add_node(1, x=34.7700, y=32.0800)
    G.add_node(2, x=34.7710, y=32.0800)   # ~94מ' מזרחה מנקודה 1 באותו קו-רוחב
    G.add_edge(1, 2, key=0, length=94.0)
    G.add_edge(2, 1, key=0, length=94.0)
    return G


def test_snap_tci_matches_nearby_edge():
    G = _tiny_graph()
    tci_by_uv = {(1, 2): 3.5}
    route_latlon = [(32.0800, 34.7700), (32.0800, 34.7710)]   # חופף כמעט לחלוטין לקשת (1,2)
    result = _snap_tci_to_latlon_path(route_latlon, tci_by_uv, G)
    assert result["avg_tci"] == 3.5
    assert result["tci_list"] == [3.5]
    assert result["high_exposure_m"] == 0.0


def test_snap_tci_high_exposure_counted():
    G = _tiny_graph()
    tci_by_uv = {(1, 2): 8.0}   # מעל HIGH_EXPOSURE_TCI (5.5)
    route_latlon = [(32.0800, 34.7700), (32.0800, 34.7710)]
    result = _snap_tci_to_latlon_path(route_latlon, tci_by_uv, G)
    assert result["avg_tci"] == 8.0
    assert result["high_exposure_m"] > 0.0


def test_snap_tci_far_segment_returns_none():
    G = _tiny_graph()
    tci_by_uv = {(1, 2): 3.5}
    # מקטע רחוק בהרבה (קילומטרים) מכל קשת בגרף — הרבה מעל _SNAP_MAX_DIST_M
    route_latlon = [(32.1200, 34.9000), (32.1201, 34.9010)]
    result = _snap_tci_to_latlon_path(route_latlon, tci_by_uv, G)
    assert result["tci_list"] == [None]
    assert result["avg_tci"] is None


def test_snap_tci_empty_path_single_point():
    G = _tiny_graph()
    result = _snap_tci_to_latlon_path([(32.08, 34.77)], {}, G)
    assert result == {"avg_tci": None, "tci_list": [], "high_exposure_m": 0.0}
