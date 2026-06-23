"""OpenStreetMap XML parser + ego-centric patch extraction.

Ported from SDTagNet's ``tools/sdtagnet/osm_parser.py`` (numpy/shapely only —
already free of mmdetection3d). The only change: the dataset-specific city
coordinate conversions are replaced by the generic
``wgs84_to_local.wgs84_to_ego_local`` projection, so nodes/ways are placed
directly in the ego (X=forward, Y=left) frame and the patch is a metric box in
that frame.

Typical use (offline, per frame):

    osm = parse(open("area.osm"))
    osm.build_node_way_lists(ego_lat, ego_lon, ego_heading)
    patch = shapely.geometry.box(x_min, y_min, x_max, y_max)   # pc_range
    elements = osm.get_elements_in_patch(patch)

``elements`` is the raw per-frame dict (lat/lon already in ego metres, tags as
strings) consumed by ``osm_tokenize.tokenize_osm_elements``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from xml.etree import cElementTree as ET

import numpy as np

from .wgs84_to_local import wgs84_to_ego_local

def iterparse(fileobj):
    context = iter(ET.iterparse(fileobj, events=("start", "end")))
    _event, root = next(context)
    return root, context


@contextmanager
def log_file_on_exception(xml):
    try:
        yield
    except SyntaxError as ex:
        import tempfile
        _fd, filename = tempfile.mkstemp('.osm')
        xml.seek(0)
        with open(filename, 'w') as f:
            f.write(xml.read())
        print('SyntaxError in xml: %s, (stored dump %s)' % (ex, filename))


@dataclass
class Node:
    id: int
    lat_lon: np.ndarray
    tags: dict


@dataclass
class Way:
    id: int
    node_lat_lon: np.ndarray
    node_ids: np.ndarray
    tags: dict


@dataclass
class RelationMember:
    id: int
    el_type: str
    role: str


@dataclass
class Relation:
    id: int
    members: list
    member_ids: np.ndarray
    tags: dict


def tag_dict_to_str(tag_dict):
    string = ""
    for key, val in tag_dict.items():
        string += key + ': ' + str(val) + ', '
    return string


def all_members_in_patch(relation, ways_patch_ids, nodes_in_patch_ids, relation_ids,
                         filter_with_relations=False):
    for id in relation.member_ids:
        if id in relation_ids and not filter_with_relations:
            continue
        elif id not in ways_patch_ids and id not in nodes_in_patch_ids and id not in relation_ids and filter_with_relations:
            return False
        elif id not in ways_patch_ids and id not in nodes_in_patch_ids and not filter_with_relations:
            return False
    return True


class OSMMapElements:
    def __init__(self, nodes=None, ways=None, relations=None):
        self.nodes = nodes if nodes is not None else dict()
        self.ways = ways if ways is not None else dict()
        self.relations = relations if relations is not None else dict()

    def build_node_way_lists(self, ego_lat, ego_lon, ego_heading):
        """Project every node/way point into the ego-local (forward, left) frame."""
        self.node_list = list(self.nodes.values())
        self.node_id_array = np.array([node.id for node in self.node_list])
        if self.node_list:
            node_ll = np.array([node.lat_lon for node in self.node_list])
            self.node_point_array_local = wgs84_to_ego_local(
                node_ll[:, 0], node_ll[:, 1], ego_lat, ego_lon, ego_heading
            )
        else:
            self.node_point_array_local = np.zeros((0, 2))

        self.way_list = list(self.ways.values())
        self.way_id_array = np.array([way.id for way in self.way_list])
        self.way_point_list_local = []
        for way in self.way_list:
            if len(way.node_lat_lon):
                self.way_point_list_local.append(
                    wgs84_to_ego_local(
                        way.node_lat_lon[:, 0], way.node_lat_lon[:, 1],
                        ego_lat, ego_lon, ego_heading,
                    )
                )
            else:
                self.way_point_list_local.append(np.zeros((0, 2)))

    def get_elements_in_patch(self, patch):
        """Extract nodes/ways/relations intersecting ``patch`` (ego-metric box)."""
        from shapely.geometry import LineString, Point

        ways_patch_intersection = [
            LineString(way).intersection(patch) if len(way) >= 2 else LineString().intersection(patch)
            for way in self.way_point_list_local
        ]
        ways_in_patch_indices = [
            i for i in range(len(self.way_list)) if not ways_patch_intersection[i].is_empty
        ]

        ways_patch_intersection = [ways_patch_intersection[i] for i in ways_in_patch_indices]
        ways_patch_tags = [
            tag_dict_to_str(self.way_list[i].tags)
            for i in ways_in_patch_indices
        ]
        ways_patch_ids = [self.way_id_array[i] for i in ways_in_patch_indices]

        ways_patch_no_multilines = []
        ways_patch_tags_no_multilines = []
        ways_patch_ids_no_multilines = []
        for id, tags, lstring in zip(ways_patch_ids, ways_patch_tags, ways_patch_intersection):
            if lstring.geom_type == 'LineString':
                ways_patch_ids_no_multilines.append(id)
                ways_patch_tags_no_multilines.append(tags)
                ways_patch_no_multilines.append(lstring)
            if lstring.geom_type == 'MultiLineString':
                for single_line in lstring.geoms:
                    ways_patch_ids_no_multilines.append(id)
                    ways_patch_tags_no_multilines.append(tags)
                    ways_patch_no_multilines.append(single_line)

        nodes_in_patch_indices = [
            i for i in range(len(self.node_list))
            if patch.contains(Point(self.node_point_array_local[i]))
        ]
        nodes_in_patch = [self.node_point_array_local[i] for i in nodes_in_patch_indices]
        nodes_in_patch_ids = [self.node_id_array[i] for i in nodes_in_patch_indices]

        relation_ids = [rel.id for rel in self.relations.values()]
        rels_1st = [
            rel for rel in self.relations.values()
            if all_members_in_patch(rel, ways_patch_ids, nodes_in_patch_ids, relation_ids)
        ]
        rels_1st_ids = [rel.id for rel in rels_1st]

        nodes_in_patch_used_indices = []
        nodes_in_patch_used_tags = []

        if rels_1st:
            rels_in_patch = [
                rel for rel in self.relations.values()
                if all_members_in_patch(rel, ways_patch_ids, nodes_in_patch_ids,
                                        rels_1st_ids, filter_with_relations=True)
            ]
            rels_in_patch_ids = [rel.id for rel in rels_in_patch]
            rels_in_patch_tags = [
                tag_dict_to_str(rel.tags) for rel in rels_in_patch
            ]
        else:
            rels_in_patch = []
            rels_in_patch_ids = []
            rels_in_patch_tags = []

        rels_node_member_indices = [list() for _ in rels_in_patch]
        rels_way_member_indices = [list() for _ in rels_in_patch]
        rels_relation_member_indices = [list() for _ in rels_in_patch]
        rels_node_member_tags = [list() for _ in rels_in_patch]
        rels_way_member_tags = [list() for _ in rels_in_patch]
        rels_relation_member_tags = [list() for _ in rels_in_patch]

        for i, rel in enumerate(rels_in_patch):
            for member in rel.members:
                if member.el_type == 'node':
                    nodes_in_patch_used_indices.append(nodes_in_patch_ids.index(member.id))
                    node_tags = self.nodes[member.id].tags
                    nodes_in_patch_used_tags.append(
                        tag_dict_to_str(node_tags) if node_tags else ""
                    )
                    rels_node_member_indices[i].append(len(nodes_in_patch_used_indices) - 1)
                    rels_node_member_tags[i].append(
                        'type: ' + member.el_type + ', role: ' + member.role + ', '
                    )
                if member.el_type == 'way':
                    related = [j for j, id in enumerate(ways_patch_ids_no_multilines) if id == member.id]
                    rels_way_member_indices[i].extend(related)
                    rels_way_member_tags[i].extend(
                        ['type: ' + member.el_type + ', role: ' + member.role + ', ' for _ in related]
                    )
                if member.el_type == 'relation':
                    rels_relation_member_indices[i].append(rels_in_patch_ids.index(member.id))
                    rels_relation_member_tags[i].append(
                        'type: ' + member.el_type + ', role: ' + member.role + ', '
                    )

        for i in range(len(nodes_in_patch_ids)):
            if i in nodes_in_patch_used_indices:
                continue
            elif self.node_list[nodes_in_patch_indices[i]].tags:
                nodes_in_patch_used_indices.append(i)
                nodes_in_patch_used_tags.append(
                    tag_dict_to_str(self.node_list[nodes_in_patch_indices[i]].tags)
                )

        nodes_in_patch_used = np.array([nodes_in_patch[i] for i in nodes_in_patch_used_indices])

        return dict(
            osm_map_nodes_pts=nodes_in_patch_used,
            osm_map_nodes_tags=nodes_in_patch_used_tags,
            osm_map_ways_pts=ways_patch_no_multilines,
            osm_map_ways_tags=ways_patch_tags_no_multilines,
            osm_map_relations_tags=rels_in_patch_tags,
            osm_map_relations_node_member_indices=rels_node_member_indices,
            osm_map_relations_way_member_indices=rels_way_member_indices,
            osm_map_relations_relation_member_indices=rels_relation_member_indices,
            osm_map_relations_node_member_tags=rels_node_member_tags,
            osm_map_relations_way_member_tags=rels_way_member_tags,
            osm_map_relations_relation_member_tags=rels_relation_member_tags,
        )


def parse(xml):
    """Parse an OSM XML file object into ``OSMMapElements`` (lat/lon retained)."""
    nodes, ways, relations = {}, {}, {}
    tags, refs, members = {}, [], []
    root, context = iterparse(xml)

    with log_file_on_exception(xml):
        for event, elem in context:
            if event == 'start':
                continue
            if elem.tag == 'tag':
                tags[elem.attrib['k']] = elem.attrib['v']
            elif elem.tag == 'node':
                osmid = int(elem.attrib['id'])
                lat, lon = float(elem.attrib['lat']), float(elem.attrib['lon'])
                nodes[osmid] = ((lat, lon), tags)
                tags = {}
            elif elem.tag == 'nd':
                refs.append(int(elem.attrib['ref']))
            elif elem.tag == 'member':
                members.append((int(elem.attrib['ref']), elem.attrib['type'], elem.attrib['role']))
            elif elem.tag == 'way':
                osm_id = int(elem.attrib['id'])
                ways[osm_id] = (osm_id, tags, refs)
                refs, tags = [], {}
            elif elem.tag == 'relation':
                osm_id = int(elem.attrib['id'])
                relations[osm_id] = (osm_id, tags, members)
                members, tags = [], {}
            root.clear()

    map_els = OSMMapElements()
    for id, node in nodes.items():
        map_els.nodes[id] = Node(id, np.array([node[0][0], node[0][1]]), node[1])
    for id, way in ways.items():
        # Some way node refs can be missing from the extract; skip those.
        coords = [map_els.nodes[nid].lat_lon for nid in way[2] if nid in map_els.nodes]
        map_els.ways[id] = Way(
            id, np.array(coords) if coords else np.zeros((0, 2)),
            np.array(way[2]), way[1],
        )
    for id, relation in relations.items():
        rel_members = [RelationMember(m[0], m[1], m[2]) for m in relation[2]]
        map_els.relations[id] = Relation(
            id, rel_members, np.array([m.id for m in rel_members]), relation[1]
        )
    return map_els
