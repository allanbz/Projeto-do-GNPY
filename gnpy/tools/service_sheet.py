#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.tools.service_sheet
========================

XLS parser that can be called to create a JSON request file in accordance with
Yang model for requesting path computation.

See: draft-ietf-teas-yang-path-computation-01.txt
"""

from xlrd import open_workbook, XL_CELL_EMPTY
from collections import namedtuple
from logging import getLogger
from copy import deepcopy
from gnpy.core.utils import db2lin
from gnpy.core.exceptions import ServiceError
from gnpy.core.elements import Transceiver, Roadm, Edfa, Fiber
import gnpy.core.ansi_escapes as ansi_escapes
from gnpy.tools.convert import corresp_names, corresp_next_node

SERVICES_COLUMN = 12


def all_rows(sheet, start=0):
    return (sheet.row(x) for x in range(start, sheet.nrows))


logger = getLogger(__name__)


class Request(namedtuple('Request', 'request_id source destination trx_type mode \
    spacing power nb_channel disjoint_from nodes_list is_loose path_bandwidth')):
    def __new__(cls, request_id, source, destination, trx_type,  mode=None, spacing=None, power=None, nb_channel=None, disjoint_from='',  nodes_list=None, is_loose='', path_bandwidth=None):
        return super().__new__(cls, request_id, source, destination, trx_type, mode, spacing, power, nb_channel, disjoint_from,  nodes_list, is_loose, path_bandwidth)


class Element:
    def __eq__(self, other):
        return type(self) == type(other) and self.uid == other.uid

    def __hash__(self):
        return hash((type(self), self.uid))


class Request_element(Element):
    def __init__(self, Request, equipment, bidir):
        # request_id is str
        # excel has automatic number formatting that adds .0 on integer values
        # the next lines recover the pure int value, assuming this .0 is unwanted
        self.request_id = correct_xlrd_int_to_str_reading(Request.request_id)
        self.source = f'trx {Request.source}'
        self.destination = f'trx {Request.destination}'
        # TODO: the automatic naming generated by excel parser requires that source and dest name
        # be a string starting with 'trx' : this is manually added here.
        self.srctpid = f'trx {Request.source}'
        self.dsttpid = f'trx {Request.destination}'
        self.bidir = bidir
        # test that trx_type belongs to eqpt_config.json
        # if not replace it with a default
        try:
            if equipment['Transceiver'][Request.trx_type]:
                self.trx_type = correct_xlrd_int_to_str_reading(Request.trx_type)
            if Request.mode is not None:
                Requestmode = correct_xlrd_int_to_str_reading(Request.mode)
                if [mode for mode in equipment['Transceiver'][Request.trx_type].mode if mode['format'] == Requestmode]:
                    self.mode = Requestmode
                else:
                    msg = f'Request Id: {self.request_id} - could not find tsp : \'{Request.trx_type}\' with mode: \'{Requestmode}\' in eqpt library \nComputation stopped.'
                    # print(msg)
                    logger.critical(msg)
                    raise ServiceError(msg)
            else:
                Requestmode = None
                self.mode = Request.mode
        except KeyError:
            msg = f'Request Id: {self.request_id} - could not find tsp : \'{Request.trx_type}\' with mode: \'{Request.mode}\' in eqpt library \nComputation stopped.'
            # print(msg)
            logger.critical(msg)
            raise ServiceError(msg)
        # excel input are in GHz and dBm
        if Request.spacing is not None:
            self.spacing = Request.spacing * 1e9
        else:
            msg = f'Request {self.request_id} missing spacing: spacing is mandatory.\ncomputation stopped'
            logger.critical(msg)
            raise ServiceError(msg)
        if Request.power is not None:
            self.power = db2lin(Request.power) * 1e-3
        else:
            self.power = None
        if Request.nb_channel is not None:
            self.nb_channel = int(Request.nb_channel)
        else:
            self.nb_channel = None

        value = correct_xlrd_int_to_str_reading(Request.disjoint_from)
        self.disjoint_from = [n for n in value.split(' | ') if value]
        self.nodes_list = []
        if Request.nodes_list:
            self.nodes_list = Request.nodes_list.split(' | ')
        self.loose = 'LOOSE'
        if Request.is_loose.lower() == 'no':
            self.loose = 'STRICT'
        self.path_bandwidth = None
        if Request.path_bandwidth is not None:
            self.path_bandwidth = Request.path_bandwidth * 1e9
        else:
            self.path_bandwidth = 0

    uid = property(lambda self: repr(self))

    @property
    def pathrequest(self):
        # Default assumption for bidir is False
        req_dictionnary = {
            'request-id': self.request_id,
            'source': self.source,
            'destination': self.destination,
            'src-tp-id': self.srctpid,
            'dst-tp-id': self.dsttpid,
            'bidirectional': self.bidir,
            'path-constraints': {
                'te-bandwidth': {
                    'technology': 'flexi-grid',
                    'trx_type': self.trx_type,
                    'trx_mode': self.mode,
                    'effective-freq-slot': [{'N': None, 'M': None}],
                    'spacing': self.spacing,
                    'max-nb-of-channel': self.nb_channel,
                    'output-power': self.power
                }
            }
        }

        if self.nodes_list:
            req_dictionnary['explicit-route-objects'] = {}
            temp = {'route-object-include-exclude': [
                {'explicit-route-usage': 'route-include-ero',
                 'index': self.nodes_list.index(node),
                 'num-unnum-hop': {
                     'node-id': f'{node}',
                     'link-tp-id': 'link-tp-id is not used',
                     'hop-type': f'{self.loose}',
                 }
                 }
                for node in self.nodes_list]
            }
            req_dictionnary['explicit-route-objects'] = temp
        if self.path_bandwidth is not None:
            req_dictionnary['path-constraints']['te-bandwidth']['path_bandwidth'] = self.path_bandwidth

        return req_dictionnary

    @property
    def pathsync(self):
        if self.disjoint_from:
            return {'synchronization-id': self.request_id,
                    'svec': {
                        'relaxable': 'false',
                        'disjointness': 'node link',
                        'request-id-number': [self.request_id] + [n for n in self.disjoint_from]
                    }
                    }
        else:
            return None
        # TO-DO: avoid multiple entries with same synchronisation vectors

    @property
    def json(self):
        return self.pathrequest, self.pathsync


def read_service_sheet(
        input_filename,
        eqpt,
        network,
        network_filename=None,
        bidir=False):
    """ converts a service sheet into a json structure
    """
    if network_filename is None:
        network_filename = input_filename
    service = parse_excel(input_filename)
    req = [Request_element(n, eqpt, bidir) for n in service]
    req = correct_xls_route_list(network_filename, network, req)
    # if there is no sync vector , do not write any synchronization
    synchro = [n.json[1] for n in req if n.json[1] is not None]
    if synchro:
        data = {
            'path-request': [n.json[0] for n in req],
            'synchronization': synchro
        }
    else:
        data = {
            'path-request': [n.json[0] for n in req]
        }
    return data


def correct_xlrd_int_to_str_reading(v):
    if not isinstance(v, str):
        value = str(int(v))
        if value.endswith('.0'):
            value = value[:-2]
    else:
        value = v
    return value


def parse_row(row, fieldnames):
    return {f: r.value for f, r in zip(fieldnames, row[0:SERVICES_COLUMN])
            if r.ctype != XL_CELL_EMPTY}


def parse_excel(input_filename):
    with open_workbook(input_filename) as wb:
        service_sheet = wb.sheet_by_name('Service')
        services = list(parse_service_sheet(service_sheet))
    return services


def parse_service_sheet(service_sheet):
    """ reads each column according to authorized fieldnames. order is not important.
    """
    logger.info(f'Validating headers on {service_sheet.name!r}')
    # add a test on field to enable the '' field case that arises when columns on the
    # right hand side are used as comments or drawing in the excel sheet
    header = [x.value.strip() for x in service_sheet.row(4)[0:SERVICES_COLUMN]
              if len(x.value.strip()) > 0]

    # create a service_fieldname independant from the excel column order
    # to be compatible with any version of the sheet
    # the following dictionnary records the excel field names and the corresponding parameter's name

    authorized_fieldnames = {
        'route id': 'request_id', 'Source': 'source', 'Destination': 'destination',
        'TRX type': 'trx_type', 'Mode': 'mode', 'System: spacing': 'spacing',
        'System: input power (dBm)': 'power', 'System: nb of channels': 'nb_channel',
        'routing: disjoint from': 'disjoint_from', 'routing: path': 'nodes_list',
        'routing: is loose?': 'is_loose', 'path bandwidth': 'path_bandwidth'}
    try:
        service_fieldnames = [authorized_fieldnames[e] for e in header]
    except KeyError:
        msg = f'Malformed header on Service sheet: {header} field not in {authorized_fieldnames}'
        logger.critical(msg)
        raise ValueError(msg)
    for row in all_rows(service_sheet, start=5):
        yield Request(**parse_row(row[0:SERVICES_COLUMN], service_fieldnames))


def correct_xls_route_list(network_filename, network, pathreqlist):
    """ prepares the format of route list of nodes to be consistant with nodes names:
        remove wrong names, find correct names for ila, roadm and fused if the entry was
        xls.
        if it was not xls, all names in list should be exact name in the network.
    """

    # first loads the base correspondance dict built with excel naming
    corresp_roadm, corresp_fused, corresp_ila = corresp_names(network_filename, network)
    # then correct dict names with names of the autodisign and find next_node name
    # according to xls naming
    corresp_ila, next_node = corresp_next_node(network, corresp_ila, corresp_roadm)
    # finally correct constraints based on these dict
    trxfibertype = [n.uid for n in network.nodes() if isinstance(n, (Transceiver, Fiber))]
    roadmtype = [n.uid for n in network.nodes() if isinstance(n, Roadm)]
    edfatype = [n.uid for n in network.nodes() if isinstance(n, Edfa)]
    # TODO there is a problem of identification of fibers in case of parallel
    # fibers between two adjacent roadms so fiber constraint is not supported
    transponders = [n.uid for n in network.nodes() if isinstance(n, Transceiver)]
    for pathreq in pathreqlist:
        # first check that source and dest are transceivers
        if pathreq.source not in transponders:
            msg = f'{ansi_escapes.red}Request: {pathreq.request_id}: could not find' +\
                f' transponder source : {pathreq.source}.{ansi_escapes.reset}'
            logger.critical(msg)
            raise ServiceError(msg)

        if pathreq.destination not in transponders:
            msg = f'{ansi_escapes.red}Request: {pathreq.request_id}: could not find' +\
                f' transponder destination: {pathreq.destination}.{ansi_escapes.reset}'
            logger.critical(msg)
            raise ServiceError(msg)
        # silently pop source and dest nodes from the list if they were added by the user as first
        # and last elem in the constraints respectively. Other positions must lead to an error
        # caught later on
        if pathreq.nodes_list and pathreq.source == pathreq.nodes_list[0]:
            pathreq.loose_list.pop(0)
            pathreq.nodes_list.pop(0)
        if pathreq.nodes_list and pathreq.destination == pathreq.nodes_list[-1]:
            pathreq.loose_list.pop(-1)
            pathreq.nodes_list.pop(-1)
        # Then process user defined constraints with respect to automatic namings
        temp = deepcopy(pathreq)
        # This needs a temporary object since we may suppress/correct elements in the list
        # during the process
        for i, n_id in enumerate(temp.nodes_list):
            # n_id must not be a transceiver and must not be a fiber (non supported, user
            # can not enter fiber names in excel)
            if n_id not in trxfibertype:
                # check that n_id is in the node list, if not find a correspondance name
                if n_id in roadmtype + edfatype:
                    nodes_suggestion = [n_id]
                else:
                    # checks first roadm, fused, and ila in this order, because ila automatic name
                    # contain roadm names. If it is a fused node, next ila names might be correct
                    # suggestions, especially if following fibers were splitted and ila names
                    # created with the name of the fused node
                    if n_id in corresp_roadm.keys():
                        nodes_suggestion = corresp_roadm[n_id]
                    elif n_id in corresp_fused.keys():
                        nodes_suggestion = corresp_fused[n_id] + corresp_ila[n_id]
                    elif n_id in corresp_ila.keys():
                        nodes_suggestion = corresp_ila[n_id]
                    else:
                        nodes_suggestion = []
                if nodes_suggestion:
                    try:
                        if len(nodes_suggestion) > 1:
                            # if there is more than one suggestion, we need to choose the direction
                            # we rely on the next node provided by the user for this purpose
                            new_n = next(n for n in nodes_suggestion
                                         if n in next_node.keys() and next_node[n]
                                         in temp.nodes_list[i:] + [pathreq.destination] and
                                         next_node[n] not in temp.nodes_list[:i])
                        else:
                            new_n = nodes_suggestion[0]
                        if new_n != n_id:
                            # warns the user when the correct name is used only in verbose mode,
                            # eg 'a' is a roadm and correct name is 'roadm a' or when there was
                            # too much ambiguity, 'b' is an ila, its name can be:
                            # Edfa0_fiber (a → b)-xx if next node is c or
                            # Edfa0_fiber (c → b)-xx if next node is a
                            msg = f'{ansi_escapes.yellow}Invalid route node specified:' +\
                                f'\n\t\'{n_id}\', replaced with \'{new_n}\'{ansi_escapes.reset}'
                            logger.info(msg)
                            pathreq.nodes_list[pathreq.nodes_list.index(n_id)] = new_n
                    except StopIteration:
                        # shall not come in this case, unless requested direction does not exist
                        msg = f'{ansi_escapes.yellow}Invalid route specified {n_id}: could' +\
                            f' not decide on direction, skipped!.\nPlease add a valid' +\
                            f' direction in constraints (next neighbour node){ansi_escapes.reset}'
                        print(msg)
                        logger.info(msg)
                        pathreq.loose_list.pop(pathreq.nodes_list.index(n_id))
                        pathreq.nodes_list.remove(n_id)
                else:
                    if temp.loose_list[i] == 'LOOSE':
                        # if no matching can be found in the network just ignore this constraint
                        # if it is a loose constraint
                        # warns the user that this node is not part of the topology
                        msg = f'{ansi_escapes.yellow}Invalid node specified:\n\t\'{n_id}\'' +\
                            f', could not use it as constraint, skipped!{ansi_escapes.reset}'
                        print(msg)
                        logger.info(msg)
                        pathreq.loose_list.pop(pathreq.nodes_list.index(n_id))
                        pathreq.nodes_list.remove(n_id)
                    else:
                        msg = f'{ansi_escapes.red}Could not find node:\n\t\'{n_id}\' in network' +\
                            f' topology. Strict constraint can not be applied.{ansi_escapes.reset}'
                        logger.critical(msg)
                        raise ServiceError(msg)
            else:
                if temp.loose_list[i] == 'LOOSE':
                    print(f'{ansi_escapes.yellow}Invalid route node specified:\n\t\'{n_id}\'' +
                          f' type is not supported as constraint with xls network input,' +
                          f' skipped!{ansi_escapes.reset}')
                    pathreq.loose_list.pop(pathreq.nodes_list.index(n_id))
                    pathreq.nodes_list.remove(n_id)
                else:
                    msg = f'{ansi_escapes.red}Invalid route node specified \n\t\'{n_id}\'' +\
                        f' type is not supported as constraint with xls network input,' +\
                        f', Strict constraint can not be applied.{ansi_escapes.reset}'
                    logger.critical(msg)
                    raise ServiceError(msg)
    return pathreqlist
