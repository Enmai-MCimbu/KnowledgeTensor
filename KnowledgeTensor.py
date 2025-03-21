"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TruthCoin Proof of Concept  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

Please run this Python 3 file. Sample data will be generated in the test directory.

"""

import os
import math
from datetime import datetime
from datetime import timedelta
import tkinter as tk
import tkinter.messagebox as msg


class MultiDimLoc:
    """
    The MultiDimLoc class supports processing of multi dimensional location information. As an example
    object/person;metric/rate_per_year;of/medical/heart_attack/subtype1;address/country1/state2/city3
    """
    separator = ';'
    hierarchy = '/'

    def __init__(self, loc_m: str):
        self.orig = loc_m
        self.lookup = {}
        self.order = loc_m.split(MultiDimLoc.separator)

        for single_loc in self.order:
            dimension = MultiDimLoc.retrieve_dimension(single_loc)
            self.lookup[dimension] = single_loc

    @staticmethod
    def retrieve_dimension(single_dim_loc: str) -> str or None:

        pos = single_dim_loc.find(MultiDimLoc.hierarchy)
        if pos == -1:
            return single_dim_loc
        else:
            return single_dim_loc[0: pos]

    @staticmethod
    def get_parent(single_dim_loc: str) -> str or None:
        if MultiDimLoc.hierarchy not in single_dim_loc:
            return None

        x = single_dim_loc.rfind(MultiDimLoc.hierarchy)
        if x == -1:
            return None
        return single_dim_loc[0: x]

    @staticmethod
    def is_root(single_dim_loc: str) -> bool:
        pos = single_dim_loc.find(MultiDimLoc.hierarchy)
        if pos == -1:
            return True
        else:
            return False

    @staticmethod
    def compute_num_levels(single_dim_loc: str) -> int:
        return single_dim_loc.count(MultiDimLoc.hierarchy)

    @staticmethod
    def build_mloc_str(parts: list) -> str:
        encountered_separator = True
        mloc_to_return = ''
        for part in parts:
            if part is None:
                mloc_to_return += MultiDimLoc.separator
                encountered_separator = True
            else:
                if encountered_separator is False:
                    mloc_to_return += MultiDimLoc.hierarchy

                mloc_to_return += part
                encountered_separator = False
        return mloc_to_return

    @staticmethod
    def filter_up_to_level(single_dim_loc: str, level: int) -> str:
        parts = single_dim_loc.split(MultiDimLoc.hierarchy)

        up_to_level = ''
        for i in range(level + 1):
            if i != 0:
                up_to_level += MultiDimLoc.hierarchy
            up_to_level += parts[i]

        return up_to_level

    def remove(self, dimension: str):
        dimension_to_use = dimension
        if MultiDimLoc.hierarchy in dimension:
            dimension_to_use = MultiDimLoc.retrieve_dimension(dimension)

        found_index = -1
        for i in range(len(self.order)):
            d = MultiDimLoc.retrieve_dimension(self.order[i])
            if d == dimension_to_use:
                found_index = i

        new_multi_dim_loc = ''
        for i in range(len(self.order)):
            if i != found_index:
                new_multi_dim_loc += self.order[i] + MultiDimLoc.separator

        new_multi_dim_loc = new_multi_dim_loc[0: len(new_multi_dim_loc) - 1]

        return MultiDimLoc(new_multi_dim_loc)

    def has_dimension(self, dimension: str) -> bool:

        return MultiDimLoc.retrieve_dimension(dimension) in self.lookup

    def __str__(self) -> str:
        return MultiDimLoc.separator.join(self.order)

    def get(self, dimension: str) -> str:
        dimension_name = MultiDimLoc.retrieve_dimension(dimension)

        if dimension_name in self.lookup:
            return self.lookup[dimension_name]
        else:
            return dimension_name


class CoreTensorRoute:
    """
    CoreTensorRoute uses the MultiDimLoc to determine where this knowledge resides in the the UniversalKnowledgeTensor.
    It needs to determine 1) which CoreTensor to go to, 2) which CoreTensorAtTime to
    go to and 3) what filter order does it support for the CoreTensorTree
    """

    def __init__(self, m_loc: MultiDimLoc):
        self.m_loc = m_loc
        self.valid_route = True

        self.core_name = CoreTensorRoute.build_core_name(self.m_loc.lookup)
        if self.core_name is None:
            self.valid_route = False

        if Dimensions.action not in self.m_loc.lookup:
            self.action = Dimensions.reality
        else:
            self.action = self.m_loc.lookup[Dimensions.action]

        self.ktp_type = CoreTensorRoute.build_ktp_type(self.m_loc.lookup)
        if self.ktp_type is None:
            self.valid_route = False

        if self.m_loc.has_dimension(Dimensions.start_time) is False:
            if self.ktp_type == KtpTypeDef.aggr or self.ktp_type == KtpTypeDef.life_score_state \
                    or self.ktp_type == KtpTypeDef.life_score_event:
                self.start_time = 0.0
            else:
                self.start_time = None
                self.valid_route = False
        else:
            self.start_time = CoreTensorRoute.parse_time(self.m_loc.lookup[Dimensions.start_time])
            if self.start_time is None:
                self.valid_route = False

        if self.m_loc.has_dimension(Dimensions.end_time) is False:
            if self.ktp_type == KtpTypeDef.aggr or self.ktp_type == KtpTypeDef.life_score_state \
                    or self.ktp_type == KtpTypeDef.life_score_event:
                self.end_time = 10000.0
            else:
                self.end_time = None
                self.valid_route = False
        else:
            self.end_time = CoreTensorRoute.parse_time(self.m_loc.lookup[Dimensions.end_time])
            if self.end_time is None:
                self.valid_route = False

        if self.m_loc.has_dimension(Dimensions.lower_bound):
            self.lower_bound = CoreTensorRoute.parse_val(self.m_loc.get(Dimensions.lower_bound))
            if self.lower_bound is None:
                self.valid_route = False
        else:
            self.valid_route = False

        if self.m_loc.has_dimension(Dimensions.upper_bound):
            self.upper_bound = CoreTensorRoute.parse_val(self.m_loc.get(Dimensions.upper_bound))
            if self.upper_bound is None:
                self.valid_route = False
        else:
            self.valid_route = False

        if self.m_loc.has_dimension(Dimensions.resolution):
            self.resolution = CoreTensorRoute.parse_val(self.m_loc.get(Dimensions.resolution))
            if self.resolution is None:
                self.valid_route = False
        else:
            self.valid_route = False

        self.filter_loc = []
        self.filter_dimension = []
        secure_hash = SecureHash()

        local_lookup = {}

        for single_dim_loc in self.m_loc.order:
            dimension = MultiDimLoc.retrieve_dimension(single_dim_loc)

            # if duplicate dimensions then it is invalid
            if dimension in local_lookup:
                self.valid_route = False
            else:
                local_lookup[dimension] = single_dim_loc

            secure_hash.update(single_dim_loc)

            if Dimensions.is_filter_dimension(dimension):
                self.filter_loc.append(single_dim_loc)
                self.filter_dimension.append(dimension)

    @staticmethod
    def build_core_name(dimension_to_single_loc: dict) -> str or None:
        s = ''

        if Dimensions.core_object not in dimension_to_single_loc:
            return None
        else:
            s += dimension_to_single_loc[Dimensions.core_object] + MultiDimLoc.separator

        if Dimensions.core_metric not in dimension_to_single_loc:
            return None
        else:
            s += dimension_to_single_loc[Dimensions.core_metric] + MultiDimLoc.separator

        if Dimensions.core_of in dimension_to_single_loc:
            s += dimension_to_single_loc[Dimensions.core_of] + MultiDimLoc.separator
        if Dimensions.core_from in dimension_to_single_loc:
            s += dimension_to_single_loc[Dimensions.core_from] + MultiDimLoc.separator

        s = s[0: len(s) - 1]
        return s

    @staticmethod
    def build_ktp_type(dimension_to_single_loc: dict) -> str or None:
        if Dimensions.ktp_type not in dimension_to_single_loc:
            return None
        else:
            ft = dimension_to_single_loc[Dimensions.ktp_type].strip()
            if ft == KtpTypeDef.dir_est:
                return KtpTypeDef.dir_est
            elif ft == KtpTypeDef.dir_meas:
                return KtpTypeDef.dir_meas
            elif ft == KtpTypeDef.aggr:
                return KtpTypeDef.aggr
            elif ft == KtpTypeDef.life_score_state:
                return KtpTypeDef.life_score_state
            elif ft == KtpTypeDef.life_score_event:
                return KtpTypeDef.life_score_event
            else:
                return None

    @staticmethod
    def is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_int(s: str) -> bool:
        try:
            int(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def parse_time(time_as_str: str):
        """
        parse_time should be able to parse a time from a single time dimension. As an example :
        start_time/2024.976  and end_time/2024/10/01/11/37/59 both need to be parsed
        """
        parts = time_as_str.split(MultiDimLoc.hierarchy)

        if len(parts) > 7:
            return None

        # We may have a situation of time/<float_val> instead of time/<YYYY>/<MM>/<DD>/<hh>/<mm>/<ss>
        if len(parts) == 2:
            s = parts[1]

            if CoreTensorRoute.is_float(s):
                return float(s)
            else:
                return None

        parts.pop(0)
        while len(parts) < 6:
            parts.append('0')

        complete_datetime = ':'.join(parts)
        t = datetime.strptime(complete_datetime, '%Y : %m : %d : %H : %M : %S')
        year = int(parts[0])

        start = datetime.strptime(str(year) + ': 1 : 1 : 0 : 0 : 0', '%Y : %m : %d : %H : %M : %S')
        end = datetime.strptime(str(year) + ': 12 : 31 : 23 : 59 : 59', '%Y : %m : %d : %H : %M : %S')

        time_span = t - start
        total_span = end - start
        numerator = time_span.total_seconds()
        denominator = total_span.total_seconds()

        return numerator / denominator + year

    @staticmethod
    def parse_val(val_as_str: str):
        """
        parse_val should be able to parse a value from a single value dimension. As an example :
        lower_bound/123.45  and upper_bound/543.21
        """

        parts = val_as_str.split(MultiDimLoc.hierarchy)

        s = parts[1]

        if CoreTensorRoute.is_float(s):
            return float(s)
        else:
            return None


class KnowledgeTensor:
    """
    KnowledgeTensor is a container for KnowledgeTensorPoints. Every expert should encode the knowledge
        into a KnowledgeTensor
    """

    def __init__(self, kt_contents: str):
        self.ktps = []

        block = KnowledgeTensor.parse(kt_contents)
        for child in block.children:
            self.ktps.append(KnowledgeTensorPoint(child))

    @staticmethod
    def parse(str_contents: str):
        block = TruthScriptBlockifier(-1, None)

        all_lines = str_contents.split('\n')

        # keep track of the active blocks. Assumes we have a maximum of 10 levels of active blocks
        active_blocks = []

        for i in range(10):
            active_blocks.append(None)

        for i in range(len(all_lines)):
            line_to_process = all_lines[i]

            if line_to_process.strip() == '':
                continue
            elif line_to_process.strip().startswith('#'):
                continue
            level = KnowledgeTensor.compute_level(line_to_process)

            block_to_process = TruthScriptBlockifier(i + 1, line_to_process)

            active_blocks[level] = block_to_process

            parent_to_add_block = block
            if level != 0:
                parent_to_add_block = active_blocks[level - 1]

            parent_to_add_block.add_child(block_to_process)
        return block

    @staticmethod
    def compute_level(line_to_process: str) -> int:
        num_spaces = 0
        for i in range(len(line_to_process)):

            letter_to_process = line_to_process[i]
            if letter_to_process == ' ':
                num_spaces += 1
            elif letter_to_process == '\t':
                num_spaces += 4
            else:
                break

        if num_spaces % 4 != 0:
            return -1
        return int(num_spaces / 4)


class KnowledgeTensorPoint:
    """
    KnowledgeTensorPoint is basic unit of knowledge. It is made up of a MultiDimLoc to specify what part of the
       UniversalKnowledgeTensor it will make a claim on. It also contains some TruthScript code to calculate the claim.
    """

    def __init__(self, truth_script_block):

        self.info = MultiDimLoc(truth_script_block.line)
        self.route = CoreTensorRoute(self.info)
        self.truth_script_block = truth_script_block
        self.parent_core_tensor = None

        sec = SecureHash()
        self.truth_script_block.hash(sec)

        hash_format = '%08x'
        self.ktp_hash_id = hash_format % sec.digest()

    def eval(self, special_stack, time_index: int):

        special_stack.push_ktp(self)

        json_node = self.parent_core_tensor.ut.json_logger.create_node()
        json_node.set_small_child('ktp_hash_id', self.ktp_hash_id)
        json_node.set_small_child('tc', self.parent_core_tensor.ut.compute_tc(self.ktp_hash_id))
        json_node.large_children_name = 'called_core_data_at_time'

        computed_value = self.truth_script_block.eval(special_stack, self, time_index)

        special_stack.pop_ktp(self)

        json_node.set_small_child('ktp_val', computed_value)
        self.parent_core_tensor.ut.json_logger.go_to_parent()

        return computed_value

    def log_to_json(self) -> None:
        json_node = self.parent_core_tensor.ut.json_logger.create_node()

        filter_dim_to_show = 'none'
        if len(self.route.filter_loc) > 0:
            filter_dim_to_show = MultiDimLoc.separator.join(self.route.filter_loc)

        json_node.set_small_child('hash_id', self.ktp_hash_id)
        json_node.set_small_child('filter_dimension', filter_dim_to_show)
        json_node.large_children_name = 'code'

        for i in range(len(self.truth_script_block.children)):
            json_node.large_children_list.append(self.truth_script_block.children[i].line)

        self.parent_core_tensor.ut.json_logger.go_to_parent()


class UniversalKnowledgeTensor:
    """
    The UniversalKnowledgeTensor contains all the knowledge from all the KnowledgeTensors.
    The UniversalKnowledgeTensor class in theory should only have a mapping of core_names to CoreTensors.
    But in practice it is a convenient container for everything used in the simulation. This includes
    HeavySeries, UserFilter, various settings, and TruthCoin counts. This allows easy access to anything
    in the simulation.
    """

    def __init__(self):
        self.user_filter = UserFilter()

        self.core_name_TO_core_tensor = {}
        self.core_name_TO_heavy_series = {}
        self.ktp_hash_id_TO_tc = {}
        self.kt_hash_id_TO_tc = {}
        self.ktp_TO_kt = {}
        self.action_TO_life_score_results = {}

        self.time_range = None
        self.last_starting_point = None
        self.completed_actions = None

        self.json_logger = None

        self.aggregate_into_distribution = False

        self.series_dependence_tracker = SeriesDependenceTracker()

        self.discount_rate = 0.98

    def compute_tc(self, ktp_hash_id: str) -> float:
        if ktp_hash_id not in self.ktp_hash_id_TO_tc:
            kt_s = self.ktp_TO_kt[ktp_hash_id]
            accumulated_tc = 0.0
            for kt in kt_s.keys():
                accumulated_tc += self.kt_hash_id_TO_tc[kt]
            self.ktp_hash_id_TO_tc[ktp_hash_id] = accumulated_tc

        return self.ktp_hash_id_TO_tc[ktp_hash_id]

    def add_ktp(self, ktp: KnowledgeTensorPoint) -> None:
        core_name = ktp.route.core_name
        if core_name not in self.core_name_TO_core_tensor:
            self.core_name_TO_core_tensor[core_name] = CoreTensor(self, core_name)

        self.core_name_TO_core_tensor[core_name].append(ktp)

    def retrieve_heavy_series(self, core_name: str):
        if core_name in self.core_name_TO_heavy_series:
            return self.core_name_TO_heavy_series[core_name]

        self.core_name_TO_heavy_series[core_name] = HeavySeries(core_name, self)
        return self.core_name_TO_heavy_series[core_name]

    def calculate_life_score(self, life_score_core_name: str, dir_for_output: str) -> None:
        self.last_starting_point = life_score_core_name

        stack = SpecialStack(self)

        life_score_heavy_series = self.retrieve_heavy_series(life_score_core_name)

        # start the simulation with reality. Action Propagation will occur later
        # to determine what other actions the user can take to improve their life_score
        stack.todo_actions[Dimensions.reality] = True

        # keep processing the todo actions until we have processed every action that was propagated
        while len(stack.todo_actions) > 0:
            todo_actions = list(stack.todo_actions.keys())
            format_template = '%03d'
            todo_actions.sort(key=lambda action: str(format_template % len(action)) + '_' + action)
            stack.active_action = todo_actions[0]

            if stack.active_action == Dimensions.reality:
                print('   calculate LifeScore for reality')
            else:
                print('   calculate LifeScore for ' + stack.active_action)

            todo = MultiDimLoc.separator.join(todo_actions)

            self.json_logger = JsonLogger()
            self.json_logger.set_small_child('todo', todo)
            self.json_logger.active_node.large_children_name = 'runs'

            accumulated_score = 0.0
            accumulated_weight = 0.0
            # run simulation at every time_index the user has specified
            for time_index in range(self.time_range.num_time_epochs()):
                json_node = self.json_logger.create_node()
                json_node.set_small_child('time_index', time_index)
                json_node.large_children_name = 'start_core_data_at_time'

                score_at_t = life_score_heavy_series.calculate(stack, time_index)

                discounted_weight = math.pow(self.discount_rate, time_index)
                accumulated_weight += discounted_weight

                if isinstance(score_at_t, float):
                    accumulated_score += score_at_t * discounted_weight
                elif isinstance(score_at_t, ProbFloatDist):
                    accumulated_score += score_at_t.calculate_mean() * discounted_weight

                self.json_logger.go_to_parent()
            self.action_TO_life_score_results[stack.active_action] = accumulated_score / accumulated_weight

            action_for_log = stack.active_action.split(MultiDimLoc.hierarchy)[-1]

            if stack.active_action == Dimensions.reality:
                action_for_log = 'reality'

            self.json_logger.log_to_json(
                os.path.join(dir_for_output, 'computation_tree_of_life_score_for_' + action_for_log + '.json'))

            stack.done_actions[stack.active_action] = True
            stack.todo_actions.pop(stack.active_action)

            self.completed_actions = stack.done_actions

    def process_block_chain(self, block_chain: list) -> None:

        awarded_tc = {}
        kt_purchases = {}

        for transaction in block_chain:
            if transaction['transaction_type'] == 'award_tc':
                if transaction['tc_holder'] not in awarded_tc:
                    awarded_tc[transaction['tc_holder']] = {}
                awarded_tc[transaction['tc_holder']][transaction['sub_foundation']] = transaction['transaction_time']
            elif transaction['transaction_type'] == 'retract_award_tc':
                awarded_tc[transaction['tc_holder']].pop(transaction['sub_foundation'])
            elif transaction['transaction_type'] == 'knowledge_tensor_purchase':
                if transaction['tc_holder'] not in kt_purchases:
                    kt_purchases[transaction['tc_holder']] = {}
                kt_purchases[transaction['tc_holder']][transaction['knowledge_tensor_hash']] = transaction[
                    'transaction_time']
            elif transaction['transaction_type'] == 'retract_knowledge_tensor_purchase':
                kt_purchases[transaction['tc_holder']].pop(transaction['knowledge_tensor_hash'])

        self.kt_hash_id_TO_tc = {}

        for tc_holder in kt_purchases.keys():
            if tc_holder not in awarded_tc or len(awarded_tc[tc_holder]) < 8:
                continue

            for kt_hash in kt_purchases[tc_holder].keys():
                if kt_hash not in self.kt_hash_id_TO_tc:
                    self.kt_hash_id_TO_tc[kt_hash] = 0.0

                self.kt_hash_id_TO_tc[kt_hash] += 1.0

    def calculate(self, special_stack, time_index: int, core_name: str):
        heavy_series = self.retrieve_heavy_series(core_name)
        return heavy_series.calculate(special_stack, time_index)

    def load_kt(self, kt_hash_id: str, kt_contents: str) -> None:

        kt = KnowledgeTensor(kt_contents)
        for ktp in kt.ktps:
            ktp_hash_id = ktp.ktp_hash_id
            if ktp_hash_id not in self.ktp_TO_kt:
                self.ktp_TO_kt[ktp_hash_id] = {}
            self.ktp_TO_kt[ktp_hash_id][kt_hash_id] = True

            self.add_ktp(ktp)

    def retrieve_all_actions(self) -> list:
        return sorted(self.completed_actions.keys())

    def log_results_to_json(self, file_name: str) -> None:
        json_logger = JsonLogger()
        json_node = json_logger.active_node
        json_node.large_children_name = ''

        for action_processed in self.retrieve_all_actions():
            json_node.set_small_child(action_processed, self.action_TO_life_score_results[action_processed])

        json_logger.log_to_json(file_name)

    def log_core_tensors_to_json(self, file_name: str) -> None:
        self.json_logger = JsonLogger()
        node = self.json_logger.active_node
        node.set_small_child('count', len(self.core_name_TO_core_tensor))
        node.large_children_name = 'core_tensors'
        core_tensor_keys = sorted(self.core_name_TO_core_tensor.keys())

        for i in range(len(core_tensor_keys)):
            self.core_name_TO_core_tensor[core_tensor_keys[i]].log_to_json()

        self.json_logger.log_to_json(file_name)


class CoreTensor:
    """
    CoreTensor contains all KnowledgeTensorPoints with the same core_name. It is a shallow container since
    it contains a simple list of CoreTensorAtTime ( 1 for every time index in the simulation).
    As an example if you want to use the UniversalKnowledgeTensor to perform a simulation from year 2020 to 2100
    you would expect, to have 80 CoreTensorAtTimes, each containing the knowledge (aka KnowledgeTensorPoints)
    applicable to that year
    """

    def __init__(self, ut: UniversalKnowledgeTensor, core_name: str):
        self.ut = ut
        self.core_name = core_name
        self.core_obj = MultiDimLoc(self.core_name).get(Dimensions.core_object)
        self.list_of_core_tensor_at_times = []

    def append(self, ktp: KnowledgeTensorPoint) -> None:
        """
        appending KnowledgeTensorPoints is a little more complicated than just appending it to
        list_of_core_tensor_at_times the simulation will have many time_indexes therefore the ktp may
        be active on multiple time_indexes. For every time_index(that the KnowledgeTensorPoint is active on),
        that ktp needs to be added to every CoreTensorAtTime. This will usually lead to the same
        KnowledgeTensorPoint being added to multiple CoreTensorAtTimes.
        """

        ktp.parent_core_tensor = self

        start_index = 0
        if ktp.route.start_time is not None:
            start_index = self.ut.time_range.index_float_to_int_round(ktp.route.start_time)

        end_index = self.ut.time_range.num_time_epochs()
        if ktp.route.end_time is not None:
            end_index = self.ut.time_range.index_float_to_int_round(ktp.route.end_time)

        start_index = max(start_index, 0)
        end_index = min(end_index, self.ut.time_range.num_time_epochs())

        for time_index in range(start_index, end_index):
            while len(self.list_of_core_tensor_at_times) <= time_index:
                self.list_of_core_tensor_at_times.append(CoreTensorAtTime(self, time_index))

            self.list_of_core_tensor_at_times[time_index].append(ktp)

    def retrieve_action_to_ktps(self, time_index: int) -> dict:
        return self.list_of_core_tensor_at_times[time_index].retrieve_ktps(self.ut.user_filter)

    def log_to_json(self) -> None:
        json_node = self.ut.json_logger.create_node()
        json_node.set_small_child('core_name', self.core_name)
        json_node.large_children_name = 'ktps'

        shown_ktps = {}
        for i in range(len(self.list_of_core_tensor_at_times)):
            self.list_of_core_tensor_at_times[i].log_to_json(shown_ktps)
        self.ut.json_logger.go_to_parent()


class CoreTensorAtTime:
    """
    CoreTensorAtTime serves a very important and complicated role in storing and retrieving KnowledgeTensorPoints.
    Storing the data occurs in 2 phases: raw and tree.
    - The raw phase is simple and the only complexity is to not store a KnowledgeTensorPoint if it already exists
    - the tree phase is where the real complexity starts. The ultimate goal is to retrieve data
      based on which filter dimensions are most important and which filter_dimensions are specified by
      the user.

      The tree phase starts the first time a get KnowledgeTensorPoint request is made using the user_filter.
      By this time all the bc processing should be complete and all the KnowledgeTensorPoint should have been added
      to the respective CoreTensorAtTime's.
      Creating the tree has 4 steps.
         1) to determine the filter order (i.e. which filter dimension is most important).
         2) create the root tree based on the most important filter dimension.
         3) give the root tree every single KnowledgeTensorPoint in the raw list
         4) push those KnowledgeTensorPoints down the tree so that the tree can create subtrees

      Getting the list of KnowledgeTensorPoints is now a matter of using the tree to get them. This class also
        adds some minor conveniences like mapping the action to the list as well as preventing a DirectAttack

    """

    def __init__(self, parent: CoreTensor, time_index: int):
        self.parent_sub_ut = parent
        self.raw_ktps = []
        self.time_index = time_index
        self.tree = None

    def append(self, ktp: KnowledgeTensorPoint) -> None:
        for raw_ktp in self.raw_ktps:
            if raw_ktp.ktp_hash_id == ktp.ktp_hash_id:
                return

        self.raw_ktps.append(ktp)

    def calculate_filter_order(self) -> list:
        """
        To determine which dimension has the highest priority, we just have to look at the KnowledgeTensorPoints
        and then the importance of each dimension is based on which dimension is specified first.
        But there is a small problem, not every KnowledgeTensorPoint will specify every dimension. As an example,
        assume that there are 10 filter dimensions that all the KnowledgeTensorPoints specify, but each individual
        KnowledgeTensorPoint only specifies at most 4 filter dimensions.

        To solve this we weight each filter dimension specified by both the TruthCoin of the KnowledgeTensorPoint
         and its position in the filter_order. In the example above, in a KnowledgeTensorPoint, the first filter
         dimension will have a position of 10, the next 9 until we get to the last one with 7.

        Step 1) Determine what are all the filter dimensions
        Step 2) go through every KnowledgeTensorPoint
        Step 2a) go through every filter dimension in the KnowledgeTensorPoint
        Step 2b) accumulate the dimension priority based on the TruthCoin count of the KnowledgeTensorPoint and the
                 filter dimension position
        Step 3) convert the dictionary of dimension priorities to a list sorted on the dimension priorities
        """
        dimension_priority = {}
        for ktp in self.raw_ktps:
            for dimension in ktp.route.filter_dimension:
                dimension_priority[dimension] = 0.0

        max_dim = len(dimension_priority)

        for ktp in self.raw_ktps:
            tc_count = ktp.parent_core_tensor.ut.compute_tc(ktp.ktp_hash_id)
            for i in range(len(ktp.route.filter_dimension)):
                filter_dimension = ktp.route.filter_dimension[i]
                position = max_dim - i
                dimension_priority[filter_dimension] += tc_count * position

        sorted_list = list(dimension_priority.keys())
        sorted_list.sort(key=lambda dimension: dimension_priority[dimension])
        return sorted_list

    def create_tree(self) -> None:
        filter_order = self.calculate_filter_order()

        self.tree = CoreTensorTree(self, None, '', 0, filter_order)
        for ktp in self.raw_ktps:
            self.tree.add_temp_ktp(ktp)
        self.tree.push_ktps_down()

    def retrieve_ktps(self, user_filter) -> dict:
        # 1) Create the Tree if needed
        if self.tree is None:
            self.create_tree()

        # 2) use the tree to get all the KnowledgeTensorPoints based on the user filter
        filtered_ktps = self.tree.retrieve_ktps(user_filter)

        # 3) organize the KnowledgeTensorPoints based on action
        action_to_ktps = {}

        for ktp in filtered_ktps:
            if ktp.route.action not in action_to_ktps:
                action_to_ktps[ktp.route.action] = []

            action_to_ktps[ktp.route.action].append(ktp)

        # 4) prevent a DirectAttack by removing KnowledgeTensorPoints that have a ktp_type different from the others
        #    As an example, if the majority of KnowledgeTensorPoints specify this is an aggregate ktp_type and an
        #    attacker specifies this is a direct ktp_type. It will disregard all the directs.
        all_actions = list(action_to_ktps.keys())
        for action in all_actions:
            self.prevent_direct_attack(action_to_ktps[action])

        return action_to_ktps

    def prevent_direct_attack(self, ktp_list: list) -> None:

        ktp_type_to_total = {}

        for ktp in ktp_list:
            tc_of_ktp = self.parent_sub_ut.ut.compute_tc(ktp.ktp_hash_id)

            if ktp.route.ktp_type not in ktp_type_to_total:
                ktp_type_to_total[ktp.route.ktp_type] = 0
            ktp_type_to_total[ktp.route.ktp_type] += tc_of_ktp

        best_ktp_type = None
        best_ktp_type_tc = -1e9

        for ktp_type in ktp_type_to_total.keys():
            ktp_type_total = ktp_type_to_total[ktp_type]
            if ktp_type_total > best_ktp_type_tc:
                best_ktp_type_tc = ktp_type_total
                best_ktp_type = ktp_type

        i = 0
        while i < len(ktp_list):
            if ktp_list[i].route.ktp_type != best_ktp_type:
                ktp_list.pop(i)
            else:
                i += 1

        ktp_list.sort(key=lambda ktp: ktp.ktp_hash_id)

    def log_to_json(self, shown_ktps: dict) -> None:
        json_node = self.parent_sub_ut.ut.json_logger.create_node()

        json_node.large_children_name = 'ktps'
        json_node.set_small_child('time_index', self.time_index)

        self.raw_ktps.sort(key=lambda ktp: ktp.ktp_hash_id)

        for i in range(len(self.raw_ktps)):
            ktp = self.raw_ktps[i]

            if ktp.ktp_hash_id in shown_ktps:
                inner_json_node = self.parent_sub_ut.ut.json_logger.create_node()

                inner_json_node.set_small_child('shown', 'true')
                inner_json_node.set_small_child('hash_id', ktp.ktp_hash_id)
                inner_json_node.set_small_child('filter_dimension', MultiDimLoc.separator.join(ktp.route.filter_loc))

                self.parent_sub_ut.ut.json_logger.go_to_parent()
            else:

                ktp.log_to_json()
                shown_ktps[ktp.ktp_hash_id] = ktp

        self.parent_sub_ut.ut.json_logger.go_to_parent()


class CoreTensorTree:
    """
    CoreTensorTree stores the KnowledgeTensorPoints in the leaf nodes. Each layer of the tree is a filterable
    dimension. As an example, the first layer can be address and the second layer ethnicity. The tree allows
    a flexible number of filterable dimensions. The ordering of the layers is based on the KnowledgeTensorPoints
    and their TruthCoin counts.

    CoreTensorTree has the final responsibility of how data gets stored and retrieved. In addition to its layered
    tree complexity it is responsible for preventing a HighResolutionOfDimensionAttack.

    When creating a CoreTensorTree there are 2 possibilities:
    1) it is a starting or intermediate node on the tree and as a result will need to pass KnowledgeTensorPoints
         and requests for KnowledgeTensorPoints down to its children nodes
    2) it is a leaf node and it simply returns the KnowledgeTensorPoints at the leaf.

    Storing KnowledgeTensorPoints is done in 2 phases since the defense of the HighResolutionOfDimensionAttack
    needs to know all the non wild card dimensions before it can process the wild card dimensions
    Phase 1) store the KnowledgeTensorPoints based on their wild card status.
    Phase 2) push the KnowledgeTensorPoints down to the children based on dimension specified
    """

    def __init__(self, parent: CoreTensorAtTime, parent_core_tensor_tree, parent_dimension_value: str, level: int,
                 filter_order: list):

        self.core_tensor_container = parent
        self.parent_tree = parent_core_tensor_tree
        self.level = level
        self.filter_order = filter_order
        self.leaf_ktps = None
        self.temp_ktps_no_wild_card = None
        self.temp_ktps_with_wild_card = None
        self.parent_dimension_value = parent_dimension_value

        # if we are a starting or intermediate node
        if filter_order is not None and len(filter_order) > 0 and self.level < len(self.filter_order):
            self.dimension = filter_order[level]
            self.children = {}
        # if we are a leaf node
        else:
            self.dimension = ''
            self.children = None

    def add_temp_ktp(self, ktp: KnowledgeTensorPoint) -> None:

        # if we are at a leaf node just store them in the leaf KnowledgeTensorPoints and exit
        if self.dimension == '':
            if self.leaf_ktps is None:
                self.leaf_ktps = []
            self.leaf_ktps.append(ktp)
            return

        dimension_value = ktp.info.get(self.dimension)
        #  store this KnowledgeTensorPoint in a temporary store based on if it has a wild card or not
        if dimension_value.endswith('*'):
            if self.temp_ktps_with_wild_card is None:
                self.temp_ktps_with_wild_card = []

            self.temp_ktps_with_wild_card.append(ktp)
        else:
            if self.temp_ktps_no_wild_card is None:
                self.temp_ktps_no_wild_card = []

            self.temp_ktps_no_wild_card.append(ktp)

    def determine_parent_chain(self) -> str:
        parent_chain = ''
        if self.parent_tree is not None:
            parent_chain += self.parent_tree.determine_parent_chain()
        parent_chain += MultiDimLoc.separator + self.parent_dimension_value
        return parent_chain

    def push_ktps_down(self) -> None:

        # if we are at a leaf node there is no more pushing the KnowledgeTensorPoints down and we just exit
        if len(self.filter_order) == 0 or self.level == len(self.filter_order):
            return

        # Add non wild card KnowledgeTensorPoints first
        if self.temp_ktps_no_wild_card is not None:
            for ktp in self.temp_ktps_no_wild_card:
                dim_val = ktp.info.get(self.dimension)

                if dim_val not in self.children:
                    self.children[dim_val] = CoreTensorTree(self.core_tensor_container, self, dim_val, self.level + 1,
                                                            self.filter_order)

                self.children[dim_val].add_temp_ktp(ktp)

        # go through each KnowledgeTensorPoint with a wild card
        #   then add it to any non wild card dimension that matches the current wild card
        #   As an example: if an attacker specifies a KnowledgeTensorPoint with an address of
        #     address/country1/state2/city3/street4 but a proper TruthCoin holder knows that the data only exists at
        #     address/country1/state2 they would put a wild card on their KnowledgeTensorPoint so that it looks like
        #     address/country1/state2* during the push_ktps_down process, the attack KnowledgeTensorPoint would be
        #     placed in address/country1/state2/city3/street4 and the proper KnowledgeTensorPoint would be placed
        #     in both address/country1/state2  AND address/country1/state2/city3/street4. This helps dilute the
        #     attacking KnowledgeTensorPoint.
        if self.temp_ktps_with_wild_card is not None:
            for ktp in self.temp_ktps_with_wild_card:
                dim_val = ktp.info.get(self.dimension)
                dim_val = dim_val[0: len(dim_val) - 1]

                if dim_val not in self.children:
                    self.children[dim_val] = CoreTensorTree(self.core_tensor_container, self, dim_val, self.level + 1,
                                                            self.filter_order)

                for child in self.children.keys():
                    if child.startswith(dim_val):
                        self.children[child].add_temp_ktp(ktp)

        for child in self.children.keys():
            self.children[child].push_ktps_down()

    def retrieve_ktps(self, user_filter) -> list:
        # if we are a leaf node just return the leaf_ktps
        if self.leaf_ktps is not None:
            return self.leaf_ktps

        # the process of getting the data starts with determining what filter dimension was specified in
        # the user_filter. Then we iteratively go up the parental chain until we find a matching child node to request
        # data from. Just because a matching child node exists, doesnt mean the KnowledgeTensorPoint retrieval will be
        # successful. Therefore if it fails we need to keep going up the tree.
        #
        # Example: the user specifies an address of address/country1/state2/city3/street4.
        #  But there is no child node at this level. So we look for a child node at address/country1/state2/city3.
        #  This node exists and we request the KnowledgeTensorPoints from it. Unfortunately, this request is
        #   unsuccessful since a data is only specified for ethnicity of ethnicity/e2 and the user is filtering on
        #   ethnicity/e1. Now we need to go up one level and look for KnowledgeTensorPoints at
        #   address/country1/state2. KnowledgeTensorPoints are found and the retrieval logic is complete

        filter_orig_val = user_filter.retrieve_filter_value(self.core_tensor_container.parent_sub_ut.core_obj,
                                                            self.dimension)
        filter_val = filter_orig_val

        while True:
            if filter_val in self.children:
                possible_ktps = self.children[filter_val].retrieve_ktps(user_filter)
                if possible_ktps is not None:
                    return possible_ktps

            if MultiDimLoc.is_root(filter_val):
                break

            filter_val = MultiDimLoc.get_parent(filter_val)


class HeavySeries:
    """
    HeavySeries is the most general container for CoreData from KnowledgeTensorPoints executions. All the
    KnowledgeTensorPoints will have the same core_name. The series is heavy because unlike a light series which
    only contains time and values, this series contains KnowledgeTensorPoints that will compute the values.
    """

    def __init__(self, core_name: str, parent_ut: UniversalKnowledgeTensor):
        self.core_name = core_name
        self.parent_ut = parent_ut
        self.list_of_core_data_at_times = []

        for i in range(self.parent_ut.time_range.num_time_epochs()):
            self.list_of_core_data_at_times.append(None)

    def calculate(self, special_stack, time_index: int):
        if self.list_of_core_data_at_times[time_index] is None:
            self.list_of_core_data_at_times[time_index] = CoreDataAtTime(self, time_index)

        return self.list_of_core_data_at_times[time_index].calculate(special_stack)


class CoreDataAtTime:
    """
    CoreDataAtTime is the main class that is responsible for execution of KnowledgeTensorPoints.
    Each HeavySeries contains 1 CoreDataAtTime for every time_index in the simulation.
    """

    def __init__(self, parent_heavy_series: HeavySeries, time_index: int):
        self.parent_heavy_series = parent_heavy_series
        self.time_index = time_index
        self.id = parent_heavy_series.core_name + MultiDimLoc.separator + str(time_index)

        self.action_to_values = {}
        self.action_to_ktps = {}
        self.propagated_actions = {}

        core_tensor = self.parent_heavy_series.parent_ut.core_name_TO_core_tensor[parent_heavy_series.core_name]

        action_to_ktps = core_tensor.retrieve_action_to_ktps(time_index)

        for action in action_to_ktps.keys():
            self.action_to_ktps[action] = action_to_ktps[action]

    def calculate(self, special_stack):
        """
        calculate is the starting point for the execution. It needs to deal with Recursion Attacks
        """

        active_action = special_stack.active_action

        special_stack.record_core_name_dependence(self.parent_heavy_series.core_name)

        # >>>>>>>>>> Recursion Attack info <<<<<<<
        # Assume the call stack points upwards.  We have the following stack
        # stack index |   stack item
        #     ------------------------
        #         8   |   core_data_at_time_2
        #         7   |   good_ktp_4
        #         6   |   core_data_at_time_4
        #         5   |   BAD_ktp_3
        #         4   |   core_data_at_time_3
        #         3   |   good_ktp_2
        #         2   |   core_data_at_time_2
        #         1   |   good_ktp_1
        #         0   |   core_data_at_time_1
        #
        # the table above can be interpreted as the user requesting:
        # core_data_at_time_1  which calls good_ktp_1 which calls core_data_at_time_2 etc...
        #
        # the stack will recognize a RecursionAttack occurring because core_data_at_time_2 exists twice.
        # And it will set its bad_ktp_index to 5 since BAD_ktp_3 has the least amount of TruthCoins. But
        # if you are core_data_at_time_4(index 6) you need to look at the bad_ktp_index and realize you
        # are above it. Therefore do not execute any TruthScript and just return None

        if special_stack.compute_last_index() >= special_stack.bad_ktp_index:
            return None

        # I am CoreTensorAtTime A. CoreTensorAtTime B already called me and I computed my value.
        # But now I am being called by CoreTensorAtTime C.
        #  There are 2 concepts
        #  1) Have I already computed the value
        #          Compute if not
        #  2) Have I already Informed the lower CoreTensorAtTime(aka caller CoreTensorAtTime) about actions it
        #         needs to consider. If I haven't, I need to inform the CoreTensorAtTime that called me of
        #         other actions it needs to consider

        if active_action not in self.action_to_values:
            json_node = self.parent_heavy_series.parent_ut.json_logger.create_node()
            json_node.set_small_child('core_data_name', self.parent_heavy_series.core_name)
            json_node.set_small_child('time_index', self.time_index)
            json_node.large_children_name = 'called_ktps'

            # Compute the value somehow. At the end of this values[active_action] will be set.
            self.compute_based_on_action_hierarchy(special_stack)

            # we were called by a Bad KnowledgeTensorPoints and need to return None
            if special_stack.compute_last_index() > special_stack.bad_ktp_index:
                json_node.set_small_child('CoreDataAtTime_EscapeBadKTP', 'todo')
                self.parent_heavy_series.parent_ut.json_logger.go_to_parent()

                return None

            json_node.set_small_child('core_data_at_time_val', self.action_to_values[active_action])
            self.parent_heavy_series.parent_ut.json_logger.go_to_parent()

        self.perform_action_propagation(special_stack)

        return self.action_to_values[active_action]

    def simple_call_ktps_and_aggregate(self, special_stack, ktps: list):
        compute_truth_coin = self.parent_heavy_series.parent_ut.compute_tc
        values = []
        truth_coins = []

        possible_lbs = {}
        possible_ubs = {}
        possible_res = {}

        for ktp in ktps:
            if ktp.route.valid_route:

                obj = ktp.eval(special_stack, self.time_index)

                if obj is not None:

                    values.append(obj)
                    truth_coin_count = compute_truth_coin(ktp.ktp_hash_id)
                    truth_coins.append(truth_coin_count)

                    lower_bound = ktp.route.lower_bound
                    if lower_bound not in possible_lbs:
                        possible_lbs[lower_bound] = 0.0
                    possible_lbs[lower_bound] += truth_coin_count

                    upper_bound = ktp.route.upper_bound
                    if upper_bound not in possible_ubs:
                        possible_ubs[upper_bound] = 0.0
                    possible_ubs[upper_bound] += truth_coin_count

                    resolution = ktp.route.resolution
                    if resolution not in possible_res:
                        possible_res[resolution] = 0.0
                    possible_res[resolution] += truth_coin_count

                    special_stack.ut.series_dependence_tracker.record_ktp_type(ktp.route.core_name, ktp.route.ktp_type,
                                                                               truth_coin_count)

        bound_check_enabled = special_stack.active_action == Dimensions.reality
        if self.parent_heavy_series.parent_ut.aggregate_into_distribution:
            return Aggregator.aggregate_into_distribution(values, truth_coins,
                                                          CoreDataAtTime.calculate_max_key(possible_lbs),
                                                          CoreDataAtTime.calculate_max_key(possible_ubs),
                                                          CoreDataAtTime.calculate_max_key(possible_res))
        else:
            return Aggregator.aggregate_single_value(values, truth_coins, bound_check_enabled,
                                                     CoreDataAtTime.calculate_max_key(possible_lbs),
                                                     CoreDataAtTime.calculate_max_key(possible_ubs),
                                                     CoreDataAtTime.calculate_max_key(possible_res))

    def compute_based_on_action_hierarchy(self, special_stack) -> None:
        """
        End Goal of this function is to set values[active_action] to something.
        We may have KnowledgeTensorPoints at action levels the same or different then the requested action level.
        This function determines if we should compute again or reuse a computation from a different action level.
        """

        attempt_push = special_stack.push_core_data_at_time(self)

        if attempt_push is False:
            # >>>>>>>>>> Recursion Attack info <<<<<<<<<<
            # We failed to Add the CoreDataAtTime to the stack since it was already there
            # Because the Push failed we are expecting bad_ktp_index to be set
            # so that all the CoreDataAtTime chain that was called by the bad_ktp return None

            return

        active_action = special_stack.active_action
        action_iterator = special_stack.active_action

        # >>>>>>>>>> Action Hierarchy <<<<<<<<<<
        # At the end of the day we need to calculate the LifeScore for every possible  action
        # But the LifeScore KnowledgeTensorPoint will usually be very simple and will not have many versions for each
        # action. You have to go many levels deep in the computation chain to encounter a KnowledgeTensorPoint which
        # will have many versions ( one for every action)
        # Therefore after encountering a KnowledgeTensorPoint with multiple action's, we need to inform the
        # UniversalKnowledgeTensor that after it is done computing the LifeScore, it needs to compute it again for
        # another action.
        #
        # But action's are hierarchical and the propagation needs to account for this.
        # As an example: lets assume we want to put solar panels on our roof .
        #      action/      : indicates our current reality without solar panels or any other action
        #      action/solar_panels_X_kw : indicates we would like to install X kw of solar on our roof
        #      action/solar_panels_X_kw/contractor_1 : indicates we would like to use contractor 1 for this job
        #      action/solar_panels_X_kw/contractor_2 : indicates we would like to use contractor 2 for this job
        #
        #  we have many KnowledgeTensorPoints all over the action tree. The size of our roof and our current
        #      electricity consumption would be at reality( aka action/).
        #  The total cost of everything ( aka LifeScore) would also be under action/ since it is a simple formula of
        #      existing recurring costs plus cost of improvements.
        #  The size of the panels and the solar panel generation would be at action/solar_panels_X_kw
        #  The cost, warranty,etc would be different between action/solar_panels_X_kw/contractor_1 and
        #          action/solar_panels_X_kw/contractor_2
        #
        #  Now let's assume during our computation chain we encounter action/solar_panels_X_kw/contractor_1.
        #  We now need to propagate action/solar_panels_X_kw/contractor_1 as well as action/solar_panels_X_kw
        #
        # Let's assume the active_action is action/solar_panels_X_kw/contractor_1
        # Given a specific active_action and a set of KnowledgeTensorPoints, there are many possible cases that
        #  need to be accounted for
        #
        #  Case 1:Compute At) Compute KnowledgeTensorPoints at Action Level: we have KnowledgeTensorPoints at this
        #         active_action and we just compute it. As an example: the solar panel cost KnowledgeTensorPoints
        #         are set in  action/solar_panels_X_kw/contractor_1
        #  Case 2:Reuse Above) Reuse Computation from Above Action Level: reuse a previous computation of these
        #         KnowledgeTensorPoints for a  different active_action. As an example: the size of the roof
        #         KnowledgeTensorPoint is specified at action/  but does not need to be recomputed for every
        #         possible contractor
        #  Case 3:Compute Above) Compute with KnowledgeTensorPoints from Above Action Level: we need to redo our
        #         KnowledgeTensorPoint computation at this  specific active_action even though we do not have
        #         KnowledgeTensorPoints at this active_action. As an example:  the LifeScore KnowledgeTensorPoint
        #         will need to be recomputed for every possible active_action even though the KnowledgeTensorPoints
        #         for the LifeScore are only specified for action/.
        #
        #    Please note that the case 2 of reusing a previous computation for the current active_action is done in
        #      calculate function and we can assume this function is only called when we do not have data for the
        #      current active_action
        #
        #   Distinguishing between Case 2:Reuse Above & Case 3:Compute Above is difficult since for both of these we
        #      do not have KnowledgeTensorPoints for this specific active_action. To solve this problem, we need to
        #      realize the  difference between  Roof Size(Case 2: Reuse Above) and LifeScore( Case 3: Compute Above ).
        #      The Roof Size KnowledgeTensorPoints do not really call many other KnowledgeTensorPoints that are likely
        #      to have many actions. Whereas the LifeScore ktps  will call many other KnowledgeTensorPoints that will
        #      have many possible actions. So when you call a ktp that does have  many possible actions, that
        #      information needs to be 'propagated' back to the calling CoreDataAtTime.
        #
        #   Now let us reconsider a more abstract example:
        #      we have an active_action of action/a1/a2/a3/a4/a5
        #          action/      (Case 1:Compute at)  level has KnowledgeTensorPoints since every KnowledgeTensorPoint
        #                          needs to be specified at action\
        #          action/a1/a2 (Case 3:Compute Above) doesn't have KnowledgeTensorPoints but it called
        #                          KnowledgeTensorPoints that propagated action at this level
        #          action/a1/a2/a3/a4 (Case 1:Compute at) has KnowledgeTensorPoints

        #      how do we determine that the other action's are Case 2:Reuse Above.
        #
        #      Now given many possible actions, what is the order of action evaluation for the LifeScore?
        #      We always start from most general to most specific. so in the above case we would compute action/ first
        #      then action/a1 then action/a1/a2 and then so on. Since every CoreDataAtTime should be defined at
        #      action/ (aka reality) we will automatically discover other actions during the computation of action/.
        #
        #      Going back to the more abstract example at every active_action computation we will do the following
        #          1) action/ : (Case 1) we compute the KnowledgeTensorPoints but the action propagation specified a
        #                       need to recompute at action/a1/a2
        #          2) action/a1: we don't have KnowledgeTensorPoints at this level( No case 1)
        #              Our options are to perform Case 2:Reuse Above OR  Case 3:Compute Above)
        #              at action/a1/a2. In this case we use Case 2. In general we never want to use data that is
        #              more specific that makes it not applicable. If you were given heart disease rates only at
        #              the Country level and city level,  which level do you use for the State level? You should
        #              use the Country level since there is more variability between cities.
        #          3) action/a1/a2: since we were informed of an action propagation, we will need to perform a
        #               (Case 3) and recompute everything even though we do not have any KnowledgeTensorPoints with
        #              action/a1/a2 only at action/
        #          4) action/a1/a2/a3: (Case 2) reuse the computation at action/a1/a2
        #          5) action/a1/a2/a3/a4: (Case 1) since we have KnowledgeTensorPoints specified
        #          6) action/a1/a2/a3/a4/a5: (Case 2) reuse the computation at action/a1/a2/a3/a4
        #
        #      Lets talk a little about Case 3:Compute Above at action/a1/a2. To perform a recomputation we need to find
        #          the KnowledgeTensorPoints to execute. There are KnowledgeTensorPoints specified at action/ and at
        #          action/a1/a2/a3/a4. We should never use anything that is more specific than our case so this leads
        #          us to use the KnowledgeTensorPoints at action/ and NOT action/a1/a2/a3/a4. Therefore we need a
        #          Case 3:Compute Above parental search that will start at action/a1/a2 and keep going up the parents
        #          until it finds a parent with KnowledgeTensorPoints.

        while action_iterator is not None:
            # Case 1) Compute At) Compute KnowledgeTensorPoints at Action Level
            if action_iterator in self.action_to_ktps:
                self.action_to_values[active_action] = self.simple_call_ktps_and_aggregate(special_stack,
                                                                                           self.action_to_ktps[
                                                                                               action_iterator])
                special_stack.pop_core_data_at_time(self)

                return

            # Case 2) Reuse a previous computation
            elif action_iterator in self.action_to_values:
                #  We need action/a1/a2/a3 to be set
                #  But we found values[action/a1] set. No KnowledgeTensorPoints or Action Propagation set in
                #   action/a1/a2 therefore we can just set values[action/a1/a2/a3] = values[action/a1]

                self.action_to_values[active_action] = self.action_to_values[action_iterator]
                special_stack.pop_core_data_at_time(self)
                return
            # Case 3) Compute Above since we detected an action propagation at this level
            elif action_iterator in self.propagated_actions:

                # First step is to initiate a Parental KnowledgeTensorPoint search. we set parental_search_action
                # to our current action_iter, and we keep going up the action tree until we find
                # a parent that has KnowledgeTensorPoints.
                # Once we find it, we evaluate the KnowledgeTensorPoints at the parent level and return

                parental_search_action = action_iterator
                while parental_search_action is not None:

                    if parental_search_action in self.action_to_ktps:
                        self.action_to_values[active_action] = self.simple_call_ktps_and_aggregate(special_stack,
                                    self.action_to_ktps[parental_search_action])
                        special_stack.pop_core_data_at_time(self)
                        return
                    else:
                        parental_search_action = MultiDimLoc.get_parent(parental_search_action)

            else:
                # No KnowledgeTensorPoints, values or action propagations were found at this level
                # we need to go up to the next level of action
                action_iterator = MultiDimLoc.get_parent(action_iterator)

    def perform_action_propagation(self, special_stack) -> None:
        """
        Action Propagation
        From a very high level perspective, a computation of a LifeScore is just
        a tree of computations of many CoreDataAtTime. The HeavySeries class
        is just a clean way of organizing the CoreDataAtTime.
        The many CoreDataAtTime are the stars of the show.
        A  CoreDataAtTime  can request data from another CoreDataAtTime  with a different
            core name OR from another CoreDataAtTime  with the same core name put at a previous time_index
            The latter is usually the case where data at time_index of 2 is a function of data at time_index of 1
            with some propagation value.
        In any case, the caller CoreDataAtTime needs to know about the action propagation.
        Fortunately, we have a SpecialStack which can let us know who the caller CoreDataAtTime is.
        But eventually, the LifeScore CoreDataAtTime will not have any CoreDataAtTime since it
        is the root computation. In this case we need to inform the UniversalKnowledgeTensor that there are other
        actions that need to be considered.
        """

        previous_core_data_at_time = special_stack.retrieve_previous_core_data_at_time()

        # If we are the first CoreDataAtTime we inform the SpecialStack which will inform the UniversalKnowledgeTensor
        if previous_core_data_at_time is None:
            for action in sorted(list(self.action_to_ktps.keys())):
                special_stack.set_todo_action(action)
            for action in sorted(list(self.propagated_actions.keys())):
                special_stack.set_todo_action(action)
        else:
            # otherwise we found a previous CodeDataAtTime and we inform them of all the actions
            # that need to be done
            for action in sorted(list(self.action_to_ktps.keys())):
                previous_core_data_at_time.add_todo_action(action)
            for action in sorted(list(self.propagated_actions.keys())):
                previous_core_data_at_time.add_todo_action(action)

    def add_todo_action(self, todo_action: str) -> None:
        if todo_action in self.action_to_values:
            return
        if todo_action in self.action_to_ktps:
            return

        self.propagated_actions[todo_action] = True

    @staticmethod
    def calculate_max_key(val: dict) -> float:
        max_key = 0
        max_amount = -100000

        for key in val.keys():
            amount = val[key]
            if amount > max_amount:
                max_amount = amount
                max_key = key

        return max_key

    def retrieve_action_to_values_float_for_gui(self) -> dict:

        float_dict = {}

        for action in self.action_to_values.keys():
            if action in self.action_to_ktps or action in self.propagated_actions:

                val = self.action_to_values[action]
                if isinstance(val, float):
                    float_dict[action] = val
                elif isinstance(val, ProbFloatDist):
                    float_dict[action] = val.calculate_mean()

        return float_dict


class SpecialStack:
    """
    SpecialStack contains an alternating stack of CoreDataAtTime and KnowledgeTensorPoints.
    It main responsibility is to perform Action Propagation and detect Recursion Attacks
    """

    active_core_datas_at_times = None

    def __init__(self, ut: UniversalKnowledgeTensor):
        self.active_action = None
        self.todo_actions = {}
        self.done_actions = {}
        self.active_core_datas_at_times = {}
        self.stack_list = []
        self.bad_ktp_index = 1000000000
        self.ut = ut

    def set_todo_action(self, new_action: str) -> None:
        if new_action in self.done_actions:
            return
        if new_action in self.todo_actions:
            return

        self.todo_actions[new_action] = True

    def push_ktp(self, ktp: KnowledgeTensorPoint) -> None:
        self.stack_list.append(ktp)

    def pop_ktp(self, ktp: KnowledgeTensorPoint) -> None:
        self.stack_list.pop(-1)

    def is_there_a_recursion_attack(self, new_core_data_at_time: CoreDataAtTime) -> bool:

        # >>>>>>>  Recursion Attack <<<<<<<<
        # In a properly functioning execution tree, no core_data_at_time would ever have itself as a parent
        # (or ancestor). So if this condition is detected, then a corrupted KnowledgeTensorPoint made its way
        # into the UniversalKnowledgeTensor. The Recursion Attack detection algorithm makes the assumption that
        # the corrupted KnowledgeTensorPoint has the lowest TruthCoin. This is somewhat a valid assumption
        # since no corrupter should ever have more tc then the actual non-corrupted TruthCoins holders. If a
        # corrupter ever gets more TruthCoins than a Recursion Attack is the least of our problems. The damage
        # a corrupter can accomplish with misinformation vastly exceeds the damage from a denial of service.
        #
        # Based on this assumption, the algorithm searches  from the bottom of the stack which has the
        # new_core_data_at_time to its ancestor and finds the KnowledgeTensorPoint with the minimum TruthCoin
        # amount. it then marks that KnowledgeTensorPoint index as the bad_ktp_index so that the CoreDataAtTime
        # and the KnowledgeTensorPoint classes can unwind the stack to remove the offending KnowledgeTensorPoints.

        # >>>> Recursion Attack Detected
        if new_core_data_at_time.id in self.active_core_datas_at_times:

            stop_index = self.active_core_datas_at_times[new_core_data_at_time.id]
            start_index = len(self.stack_list) - 1

            min_tc = 100000000.0
            min_ktp_index = -1

            for i in range(start_index, stop_index, -2):
                ktp = self.stack_list[i - 1]

                tc_of_ktp = self.ut.compute_tc(ktp.ktp_hash_id)

                if tc_of_ktp < min_tc:
                    min_tc = tc_of_ktp
                    min_ktp_index = i - 1

            self.bad_ktp_index = min_ktp_index

            return True
        else:
            return False

    def push_core_data_at_time(self, core_data_at_time: CoreDataAtTime) -> bool:

        if self.is_there_a_recursion_attack(core_data_at_time):
            return False

        self.active_core_datas_at_times[core_data_at_time.id] = len(self.stack_list)
        self.stack_list.append(core_data_at_time)

        return True

    def pop_core_data_at_time(self, core_data_at_time: CoreDataAtTime) -> None:
        self.stack_list.pop(-1)

        self.active_core_datas_at_times.pop(core_data_at_time.id)

    def retrieve_previous_core_data_at_time(self) -> CoreDataAtTime or None:
        index = len(self.stack_list) - 1

        while index >= 0:
            if isinstance(self.stack_list[index], CoreDataAtTime):
                return self.stack_list[index]

            index -= 1

        return None

    def retrieve_previous_core_data_at_time_id(self) -> str:
        previous_core_data_at_time = self.retrieve_previous_core_data_at_time()
        if previous_core_data_at_time is None:
            return ''
        else:
            return previous_core_data_at_time.id

    def compute_last_index(self) -> int:
        return len(self.stack_list) - 1

    def calculate(self, time_index: int, core_name: str):
        return self.ut.calculate(self, time_index, core_name)

    def record_core_name_dependence(self, active_core_name: str) -> None:
        # this section captures dependence information. It is not necessary for UniversalKnowledgeTensor computations,
        # but it is used by the CoreViewer to show dependence of the HeavySeries's.

        previous_core_data = self.retrieve_previous_core_data_at_time()

        if previous_core_data is None:
            return

        output_core_name = previous_core_data.parent_heavy_series.core_name
        input_core_name = active_core_name

        self.ut.series_dependence_tracker.record_dependence(output_core_name, input_core_name)


class Dimensions:
    """
    Dimensions contains a definition for all the dimensions a MultiDimLoc can have.
    The final symbol of each dimension may be up for discussion. If it changes,
    the Dimensions class provides an easy way to change it
    """

    core_object = 'object'
    core_metric = 'metric'
    core_of = 'of'
    core_from = 'from'

    action = 'action'
    # action/some_hypothetical_action_1 implies that we are interested in a simulation if
    # we were to implement some_hypothetical_action_1 like taking medication for heart disease.
    # If you have nothing after action/  then we are dealing with reality (aka no action taken)
    reality = 'action'

    ktp_type = 'ktp_type'
    start_time = 'start_time'
    end_time = 'end_time'
    lower_bound = 'lower_bound'
    upper_bound = 'upper_bound'
    resolution = 'resolution'

    address = 'address'
    ethnicity = 'ethnicity'
    gender = 'gender'
    profession = 'profession'

    smoker = 'smoker'
    genetic = 'genetic'

    @staticmethod
    def is_filter_dimension(d: str) -> bool:
        if d == Dimensions.core_object:
            return False
        elif d == Dimensions.core_metric:
            return False
        elif d == Dimensions.core_of:
            return False
        elif d == Dimensions.core_from:
            return False
        elif d == Dimensions.action:
            return False
        elif d == Dimensions.ktp_type:
            return False
        elif d == Dimensions.start_time:
            return False
        elif d == Dimensions.end_time:
            return False
        elif d == Dimensions.lower_bound:
            return False
        elif d == Dimensions.upper_bound:
            return False
        elif d == Dimensions.resolution:
            return False
        else:
            return True


class Objects:
    """
    Objects contains a definition for all the Objects the Object dimension can have.
    """
    person = Dimensions.core_object + MultiDimLoc.hierarchy + 'person'
    alien = Dimensions.core_object + MultiDimLoc.hierarchy + 'alien'
    economy = Dimensions.core_object + MultiDimLoc.hierarchy + 'economy'

    environment = Dimensions.core_object + MultiDimLoc.hierarchy + 'environment'
    city = Dimensions.core_object + MultiDimLoc.hierarchy + 'city'
    state = Dimensions.core_object + MultiDimLoc.hierarchy + 'state'
    country = Dimensions.core_object + MultiDimLoc.hierarchy + 'country'


class MetricDef:
    """
    MetricDef contains a definition for all the metrics the Metric dimension can have.
    """

    death_converter = Dimensions.core_metric + MultiDimLoc.hierarchy + 'death_converter'
    money_converter = Dimensions.core_metric + MultiDimLoc.hierarchy + 'money_converter'
    time_converter = Dimensions.core_metric + MultiDimLoc.hierarchy + 'time_converter'

    life_score_per_year = Dimensions.core_metric + MultiDimLoc.hierarchy + 'life_score_per_year'
    money_per_year = Dimensions.core_metric + MultiDimLoc.hierarchy + 'money_per_year'
    time_per_year = Dimensions.core_metric + MultiDimLoc.hierarchy + 'time_per_year'
    death_per_year = Dimensions.core_metric + MultiDimLoc.hierarchy + 'death_per_year'
    base_line = Dimensions.core_metric + MultiDimLoc.hierarchy + 'base_line'

    number_per_year = Dimensions.core_metric + MultiDimLoc.hierarchy + 'number_per_year'
    money_per_event = Dimensions.core_metric + MultiDimLoc.hierarchy + 'money_per_event'
    time_per_event = Dimensions.core_metric + MultiDimLoc.hierarchy + 'time_per_event'
    death_per_event = Dimensions.core_metric + MultiDimLoc.hierarchy + 'death_per_event'

    hours_per_year = Dimensions.core_metric + MultiDimLoc.hierarchy + 'hours_per_year'
    money_per_hour = Dimensions.core_metric + MultiDimLoc.hierarchy + 'money_per_hour'
    cpi = Dimensions.core_metric + MultiDimLoc.hierarchy + 'cpi'

    glacier_mass = Dimensions.core_metric + MultiDimLoc.hierarchy + 'glacier_mass'
    surface_water = Dimensions.core_metric + MultiDimLoc.hierarchy + 'surface_water'
    ocean_temp = Dimensions.core_metric + MultiDimLoc.hierarchy + 'ocean_temp'
    air_temp = Dimensions.core_metric + MultiDimLoc.hierarchy + 'air_temp'

    climate_change = Dimensions.core_metric + MultiDimLoc.hierarchy + 'climate_change'
    blood_pressure = Dimensions.core_metric + MultiDimLoc.hierarchy + 'blood_pressure'
    bmi = Dimensions.core_metric + MultiDimLoc.hierarchy + 'bmi'


class From:
    """
    From contains a definition for all the from's the From dimension can have.
    """

    employment_income = Dimensions.core_from + '/income/employment'
    drowning_accidents = Dimensions.core_from + '/income/accidents/drowning'
    shark_attack_employment = Dimensions.core_from + '/income/accidents/shark_attack'

    earth_quake = Dimensions.core_from + '/nature/mass_disaster/earth_quake'
    flood = Dimensions.core_from + '/nature/mass_disaster/flood'
    hurricane = Dimensions.core_from + '/nature/mass_disaster/hurricane'
    tornado = Dimensions.core_from + '/nature/mass_disaster/tornado'

    lightning_strike = Dimensions.core_from + '/nature/individual_disaster/lightning_strike'
    shark_attack_nature = Dimensions.core_from + '/nature/individual_disaster/shark_attack'

    food = Dimensions.core_from + '/food'
    cancer = Dimensions.core_from + '/medical/cancer'
    heart_attack = Dimensions.core_from + '/medical/heart_attack'
    diabetes = Dimensions.core_from + '/medical/diabetes'
    flu = Dimensions.core_from + '/medical/flu'
    covid = Dimensions.core_from + '/medical/covid'
    med_insurance = Dimensions.core_from + '/medical/insurance'
    med_tax = Dimensions.core_from + '/medical/medical_tax'

    rent = Dimensions.core_from + '/shelter/rent'
    electricity = Dimensions.core_from + '/shelter/electricity'
    water = Dimensions.core_from + '/shelter/water'
    renter_insurance = Dimensions.core_from + '/shelter/renter_insurance'

    house_fire = Dimensions.core_from + '/shelter/house_fire'
    pipe_burst = Dimensions.core_from + '/shelter/pipe_burst'

    shelter_tax = Dimensions.core_from + '/shelter/housing_tax'

    car_tran = Dimensions.core_from + '/transportation/car'
    car_accident = Dimensions.core_from + '/transportation/car_accident'
    transportation_tax = Dimensions.core_from + '/transportation/tran_tax'

    car_maintenance = Dimensions.core_from + '/transportation/car_maintenance'
    car_fuel = Dimensions.core_from + '/transportation/car_fuel'


class Of:
    """
    Of contains a definition for all the Ofs the Of dimension can have.
    """

    emp_inc = Dimensions.core_of + MultiDimLoc.hierarchy + 'income/employment'
    emp_tax = Dimensions.core_of + MultiDimLoc.hierarchy + 'income/tax'
    food_dir = Dimensions.core_of + MultiDimLoc.hierarchy + 'food' + MultiDimLoc.hierarchy + 'direct'
    food_tax = Dimensions.core_of + MultiDimLoc.hierarchy + 'food' + MultiDimLoc.hierarchy + 'tax'

    food = Dimensions.core_of + MultiDimLoc.hierarchy + 'food'


class KtpTypeDef:
    """
    KtpTypeDef contains a definition for all ktp types a ktp can have.
    """
    life_score_state = Dimensions.ktp_type + MultiDimLoc.hierarchy + 'state'
    life_score_event = Dimensions.ktp_type + MultiDimLoc.hierarchy + 'event'
    dir_est = Dimensions.ktp_type + MultiDimLoc.hierarchy + 'dir_est'
    dir_meas = Dimensions.ktp_type + MultiDimLoc.hierarchy + 'dir_meas'
    aggr = Dimensions.ktp_type + MultiDimLoc.hierarchy + 'aggregate'

    dir_mul = Dimensions.ktp_type + MultiDimLoc.hierarchy + 'dir_mul'


class Name:

    @staticmethod
    def helper(single_dim_loc: str) -> str:
        words = single_dim_loc.split(MultiDimLoc.hierarchy)
        last_word = words[-1]
        if CoreTensorRoute.is_int(last_word):
            return (words[-2] + ' category ' + last_word).replace('_', ' ')
        else:
            return last_word.replace('_', ' ')

    @staticmethod
    def create_english_description(core_name: str) -> str:
        english_description = ''
        mloc = MultiDimLoc(core_name)

        if mloc.get(Dimensions.core_object) != Objects.person:
            english_description += Name.helper(mloc.get(Dimensions.core_object)) + ' '

        english_description += Name.helper(mloc.get(Dimensions.core_metric)) + ' '

        if mloc.has_dimension(Dimensions.core_of):
            english_description += 'of ' + Name.helper(mloc.get(Dimensions.core_of)) + ' '

        if mloc.has_dimension(Dimensions.core_from):
            english_description += 'from ' + Name.helper(mloc.get(Dimensions.core_from)) + ' '

        return english_description

    @staticmethod
    def persons_metric(metric: str) -> str:
        return Objects.person + MultiDimLoc.separator + metric

    @staticmethod
    def persons_annual_money_of(of: str) -> str:
        return Objects.person + MultiDimLoc.separator + MetricDef.money_per_year + MultiDimLoc.separator + of

    @staticmethod
    def person_life_score_from(from_source: str) -> str:
        return Objects.person + MultiDimLoc.separator + MetricDef.life_score_per_year + MultiDimLoc.separator \
               + from_source

    @staticmethod
    def economic_metric_of(m: str, of: str) -> str:
        return Objects.economy + MultiDimLoc.separator + m + MultiDimLoc.separator + of

    @staticmethod
    def economic_cpi_of(of: str) -> str:
        return Name.economic_metric_of(MetricDef.cpi, of)

    @staticmethod
    def environment_metric(metric: str) -> str:
        return Objects.environment + MultiDimLoc.separator + metric

    @staticmethod
    def mloc(slocs: list) -> str:
        return MultiDimLoc.separator.join(slocs)


class TimeRange:
    """
    TimeRange is used by many classes to keep track of when the simulation will start, end, the current time and
    the resolution of the time_index. As an example, you can specify you want a simulation that starts at
    year 2020 and goes to 2100 with a present time of 2024. The resolution of time increments is 1 year.
    """

    def __init__(self, start: float, end: float, resolution: float, present_time: float):
        self.start = start
        self.end = end
        self.resolution = resolution
        self.present_time = present_time

    def num_time_epochs(self) -> int:
        delta = self.end - self.start
        num = delta / self.resolution
        return int(math.ceil(num))

    def index_int_to_float(self, int_index: int) -> float:
        return float(self.start + int_index * self.resolution)

    def index_float_to_int_round(self, float_index: float) -> int:
        f0 = float_index - self.start
        f1 = f0 * self.resolution
        f2 = round(f1)
        f3 = f2 / self.resolution
        return int(f3)


class UserFilter:
    """
    UserFilter allows the the user to specify what to filter in the filterable dimensions.
    As an example they can specify the address of address/country1/state2/city3 so that
    the UniversalKnowledgeTensor retrieves data most relevant to that address.
    """

    def __init__(self):
        self.birthday = 0
        self.object_dimension_value = {}

    def set(self, obj: str, val: str) -> None:
        if obj not in self.object_dimension_value:
            self.object_dimension_value[obj] = {}

        dimension = MultiDimLoc.retrieve_dimension(val)
        self.object_dimension_value[obj][dimension] = val

    def retrieve_filter_value(self, obj: str, filter_dimension: str) -> str:
        if obj in self.object_dimension_value:

            if filter_dimension in self.object_dimension_value[obj]:
                return self.object_dimension_value[obj][filter_dimension]

        if filter_dimension == Dimensions.address:
            if Objects.person in self.object_dimension_value and \
                    filter_dimension in self.object_dimension_value[Objects.person]:
                return self.object_dimension_value[Objects.person][filter_dimension]

        return filter_dimension

    def log_to_json(self, file_name: str) -> None:
        json_logger = JsonLogger()
        json_node = json_logger.active_node
        json_node.large_children_name = 'object_filters'

        objects = sorted(list(self.object_dimension_value.keys()))
        json_node.set_small_child('num_filter_objects', len(objects))

        for obj_index in range(len(objects)):
            inner_json_node = json_logger.create_node()

            inner_json_node.set_small_child('obj', objects[obj_index])
            inner_json_node.large_children_name = 'filters'

            dims = sorted(list(self.object_dimension_value[objects[obj_index]].keys()))
            for i in range(len(dims)):
                inner_json_node.large_children_list.append(self.object_dimension_value[objects[obj_index]][dims[i]])

            json_logger.go_to_parent()

        json_logger.log_to_json(file_name)


class ProbFloatDist:
    """
     ProbFloatDist is a probability distribution for all the values a CoreDataAtTime can have.
     As an example, you cannot perfectly predict the heart attack rate for people with characteristics
     similar to me. The data produces a distribution with different probabilities for each rate.
    """

    def __init__(self, lower_bound: float, upper_bound: float, resolution: float):

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if lower_bound > upper_bound:
            self.lower_bound = upper_bound
            self.upper_bound = lower_bound
        self.resolution = resolution

        self.lower_bound = round(self.lower_bound / resolution) * resolution
        self.upper_bound = round(self.upper_bound / resolution) * resolution

        num_items = int(math.ceil((self.upper_bound - self.lower_bound) / resolution)) + 1

        self.values = []
        for i in range(num_items):
            self.values.append(0.0)

    @staticmethod
    def uniform_distribution(uniform_start: float, uniform_end: float, lower_bound: float, upper_bound: float,
                             resolution: float):
        uniform_dist = ProbFloatDist(lower_bound, upper_bound, resolution)

        uniform_start = round(uniform_start / resolution) * resolution
        uniform_end = round(uniform_end / resolution) * resolution

        start_index = uniform_dist.index_float_to_int(uniform_start)
        end_index = uniform_dist.index_float_to_int(uniform_end)
        num_items = end_index - start_index + 1

        average_per_bin = 1.0 / num_items

        for i in range(start_index, end_index + 1):
            uniform_dist.values[i] = average_per_bin

        return uniform_dist

    def index_int_to_float(self, int_index: int) -> float:
        return self.lower_bound + int_index * self.resolution

    def index_float_to_int(self, float_index: float) -> int:
        delta = float_index - self.lower_bound
        float_int_index1 = delta / self.resolution
        float_int_index2 = math.floor(float_int_index1)

        return int(float_int_index2)

    def calculate_mean(self) -> float:
        accumulated_values = 0

        accumulated_weights = 0

        for i in range(len(self.values)):
            float_index = self.index_int_to_float(i)
            weight = self.values[i]

            accumulated_values += float_index * weight

            accumulated_weights += weight

        return accumulated_values / accumulated_weights

    def format_distribution(self, lower_bound: float, upper_bound: float, resolution: float):
        """
        During the aggregation process we may encounter a distribution that doesn't match
        the lower_bound, upper_bound and resolution that the majority of KnowledgeTensorPoints have specified.
        In this case, the distribution needs to be formatted to be in line with what the KnowledgeTensorPoints
        have specified
        """

        formatted_distribution = ProbFloatDist(lower_bound, upper_bound, resolution)

        num_below = 0
        num_above = 0

        for i in range(len(self.values)):
            index_float_old = self.index_int_to_float(i)
            prob = self.values[i]

            index_int_new = formatted_distribution.index_float_to_int(index_float_old)

            if index_int_new < 0:
                num_below += 1
            elif index_int_new >= len(formatted_distribution.values):
                num_below += 1
            else:
                formatted_distribution.values[index_int_new] += prob

        if formatted_distribution.accumulate_probabilities() == 0:
            if num_below > 0 and num_above == 0:
                formatted_distribution.values[0] = 1.0
            elif num_below == 0 and num_above > 0:
                formatted_distribution.values[-1] = 1.0

        formatted_distribution.normalize()
        return formatted_distribution

    def multiply(self, val: float):
        multiplied_distribution = ProbFloatDist(self.lower_bound * val, self.upper_bound * val, self.resolution * val)

        for i in range(len(self.values)):
            index_float_old = self.index_int_to_float(i)
            prob = self.values[i]

            multiplied_index_float_old = index_float_old * val

            index_int_new = multiplied_distribution.index_float_to_int(multiplied_index_float_old)

            if 0 <= index_int_new < len(multiplied_distribution.values):
                multiplied_distribution.values[index_int_new] += prob

        return multiplied_distribution

    def transform(self, float_index_transform, new_res: float):
        transformed_distribution = ProbFloatDist(float_index_transform(self.lower_bound),
                                                 float_index_transform(self.upper_bound), new_res)

        for i in range(len(self.values)):
            index_float_old = self.index_int_to_float(i)
            prob = self.values[i]

            transformed_index_float_old = float_index_transform(index_float_old)

            index_int_new = transformed_distribution.index_float_to_int(transformed_index_float_old)

            if index_int_new == -1:
                transformed_distribution.values[0] += prob
            elif index_int_new == len(self.values):
                transformed_distribution.values[-1] += prob

            if 0 <= index_int_new < len(transformed_distribution.values):
                transformed_distribution.values[index_int_new] += prob

        return transformed_distribution

    def clone(self):
        c = ProbFloatDist(self.lower_bound, self.upper_bound, self.resolution)

        for i in range(len(self.values)):
            c.values[i] = self.values[i]

        return c

    def accumulate_probabilities(self) -> float:
        accumulation = 0.0
        for i in range(len(self.values)):
            accumulation += self.values[i]
        return accumulation

    def normalize(self) -> None:
        sum_of_probabilities = self.accumulate_probabilities()

        if sum_of_probabilities == 1.0:
            return

        normalize_factor = 1.0 / sum_of_probabilities

        for i in range(len(self.values)):
            self.values[i] = self.values[i] * normalize_factor


class FloatAggregator:
    """
    On any given topic, multiple experts will have different opinions with different TruthCoins for each opinion.
    FloatAggregator will aggregate all these opinion into a single value
    """

    def __init__(self):
        self.values = []
        self.truth_coin_counts = []

    def append(self, float_value: float, truth_coin_count: float) -> None:
        self.truth_coin_counts.append(truth_coin_count)
        self.values.append(float_value)

    def __len__(self) -> int:
        return len(self.values)

    def aggregate(self):
        if len(self.values) == 0:
            return None

        accumulated_value = 0.0
        accumulated_weight = 0.0
        for i in range(len(self.values)):
            weight = self.truth_coin_counts[i]
            val = self.values[i]
            accumulated_value += val * weight
            accumulated_weight += weight

        return accumulated_value / accumulated_weight


class FloatDistAggregator:
    """
    FloatDistAggregator is similar to the FloatAggregator but when the UniversalKnowledgeTensor is run in distribution
    mode it aggregates the distributions
    """

    def __init__(self):
        self.float_dists = []
        self.truth_coin_counts = []

    def append(self, float_distribution: ProbFloatDist, truth_coin_count: float) -> None:
        self.truth_coin_counts.append(truth_coin_count)
        self.float_dists.append(float_distribution)

    def __len__(self) -> int:
        return len(self.float_dists)

    def aggregate(self):

        first_dist = self.float_dists[0]

        aggregate_dist = ProbFloatDist(first_dist.lower_bound, first_dist.upper_bound, first_dist.resolution)

        for bin_index in range(len(aggregate_dist.values)):

            accumulated_value = 0.0
            accumulated_weight = 0.0

            for i in range(len(self.float_dists)):
                weight = self.truth_coin_counts[i]
                val = self.float_dists[i].values[bin_index]
                accumulated_value += val * weight
                accumulated_weight += weight

            val_to_set = accumulated_value / accumulated_weight
            aggregate_dist.values[bin_index] = val_to_set

        return aggregate_dist


class Aggregator:
    """
    Aggregator is the entry point for all aggregation activity. It will look at the data to aggregate
    and based on the types of data and simulation mode, it will choose the proper aggregator.
    """

    @staticmethod
    def aggregate_single_value(values: list, truth_coin_counts: list, bound_check_enabled: bool, lower_bound: float,
                               upper_bound: float, resolution: float):
        float_aggregator = FloatAggregator()

        for i in range(len(values)):
            value = values[i]

            truth_coins = truth_coin_counts[i]

            if isinstance(value, float):
                if bound_check_enabled:
                    if value < lower_bound:
                        value = lower_bound
                    if value > upper_bound:
                        value = upper_bound

                float_aggregator.append(value, truth_coins)
            elif isinstance(value, ProbFloatDist):
                distribution = value
                mean = distribution.calculate_mean()
                float_aggregator.append(mean, truth_coins)

        return float_aggregator.aggregate()

    @staticmethod
    def aggregate_into_distribution(values: list, truth_coin_counts: list, lower_bound: float, upper_bound: float,
                                    resolution: float):

        float_aggregator = FloatAggregator()
        float_dist_aggregator = FloatDistAggregator()

        for i in range(len(values)):
            value = values[i]
            truth_coins = truth_coin_counts[i]

            if isinstance(value, float):
                float_aggregator.append(value, truth_coins)
            elif isinstance(value, ProbFloatDist):
                dist = value
                float_dist_aggregator.append(dist.format_distribution(lower_bound, upper_bound, resolution),
                                             truth_coins)

        if len(float_aggregator) > 0 and len(float_dist_aggregator) == 0:
            return float_aggregator.aggregate()
        elif len(float_aggregator) == 0 and len(float_dist_aggregator) > 0:

            return float_dist_aggregator.aggregate()


class SeriesDependenceTracker:
    """
    SeriesDependenceTracker tracks which HeavySeries is an input to other HeavySeries
    This is not used for any computation, but the CoreViewer uses it to determine what
    are the input children that can be shown.
    """

    def __init__(self):
        self.inputs_of_core_name = {}
        self.outputs_of_core_name = {}
        self.core_name_to_formula_type_counts = {}

    def record_dependence(self, core_name: str, input_core_name: str) -> None:
        if core_name not in self.inputs_of_core_name:
            self.inputs_of_core_name[core_name] = {}
        self.inputs_of_core_name[core_name][input_core_name] = True

        if input_core_name not in self.outputs_of_core_name:
            self.outputs_of_core_name[input_core_name] = {}
        self.outputs_of_core_name[input_core_name][core_name] = True

    def compute_max_formula_type(self, main_core_name: str) -> str:
        max_key = None
        max_val = -1e9

        for core_name in self.core_name_to_formula_type_counts[main_core_name].keys():
            val = self.core_name_to_formula_type_counts[main_core_name][core_name]
            if val > max_val:
                max_val = val
                max_key = core_name

        return max_key

    def record_ktp_type(self, core_name: str, formula_type: str, tc_count: float) -> None:
        if core_name not in self.core_name_to_formula_type_counts:
            self.core_name_to_formula_type_counts[core_name] = {}

        formula_type_counts = self.core_name_to_formula_type_counts[core_name]
        if formula_type not in formula_type_counts:
            formula_type_counts[formula_type] = 0

        formula_type_counts[formula_type] = formula_type_counts[formula_type] + tc_count

    def get_parent_chain(self, destination_core_name: str) -> list:
        parent_chain = []
        parent_chain.insert(0, destination_core_name)

        iterator = destination_core_name

        while iterator in self.outputs_of_core_name:
            all_outputs = list(self.outputs_of_core_name[iterator].keys())
            first_output = all_outputs[0]

            parent_chain.insert(0, first_output)
            iterator = first_output

        return parent_chain


class Operations:
    """
    Operations is central class to define what the addition, multiplication,etc.. operations do
    among floats, ProbFloatDist and other type expansions.
    """

    @staticmethod
    def addition_operation(left_object, right_object):

        if isinstance(left_object, float) and isinstance(right_object, float):
            return left_object + right_object
        elif isinstance(left_object, float) and isinstance(right_object, ProbFloatDist):
            dist_to_return = right_object.transform(lambda float_index: left_object + float_index,
                                                    right_object.resolution)
            return dist_to_return
        elif isinstance(left_object, ProbFloatDist) and isinstance(right_object, float):

            dist_to_return = left_object.transform(lambda float_index: float_index + right_object,
                                                   left_object.resolution)
            return dist_to_return

    @staticmethod
    def subtraction_operation(left_object, right_object):
        if isinstance(left_object, float) and isinstance(right_object, float):
            return left_object - right_object
        elif isinstance(left_object, float) and isinstance(right_object, ProbFloatDist):
            dist_to_return = right_object.transform(lambda float_index: left_object - float_index,
                                                    right_object.resolution)
            return dist_to_return

    @staticmethod
    def multiplication_operation(left_object, right_object):
        if isinstance(left_object, float) and isinstance(right_object, float):
            return left_object * right_object
        elif isinstance(left_object, float) and isinstance(right_object, ProbFloatDist):
            dist_to_return = right_object.transform(lambda float_index: left_object * float_index,
                                                    right_object.resolution * left_object)
            return dist_to_return

    @staticmethod
    def division_operation(left_object, right_object):
        if isinstance(left_object, float) and isinstance(right_object, float):
            return left_object / right_object

    @staticmethod
    def power_operation(left_object, right_val: float):
        if isinstance(left_object, float):
            return math.pow(left_object, right_val)


class TruthScriptBlockifier:
    """
    TruthScriptBlockifier breaks a KnowledgeTensor file into blocks of code similar to how Python breaks code
    into blocks
    """

    def __init__(self, line_number: int, line: str or None):
        self.line_number = line_number
        self.line = line
        self.parent_block = None
        self.children = []

    def add_child(self, child) -> None:
        self.children.append(child)
        child.parent_block = self

    def hash(self, sh) -> None:
        sh.update(self.line)

        for child in self.children:
            child.hash(sh)

    def eval(self, special_stack: SpecialStack, calling_ktp: KnowledgeTensorPoint, time_index: int):
        variable_lookups = {'time_index': time_index}

        for i in range(len(self.children)):
            block_to_process = self.children[i]
            line_to_process = block_to_process.line.strip()

            if '#' in line_to_process:
                line_to_process = line_to_process.split('#')[0]

            assignment = TruthScriptBlockifier.attempt_split_assignment(line_to_process)

            if assignment is not None:
                variable_name = assignment[0]
                expr = assignment[1]

                exp = TruthScriptTokenizer.tokenize(expr, variable_lookups)
                variable_lookups[variable_name] = exp.execute_operation(special_stack, calling_ktp, variable_lookups,
                                                                        None)
            elif line_to_process.startswith('return'):
                pos = line_to_process.find('return') + len('return')
                expr = line_to_process[pos:]
                exp = TruthScriptTokenizer.tokenize(expr, variable_lookups)
                return exp.execute_operation(special_stack, calling_ktp, variable_lookups, None)

    def has_str(self, str_to_look_for) -> bool:
        if str_to_look_for in self.line:
            return True

        for child in self.children:
            if child.has_str(str_to_look_for):
                return True
        return False

    @staticmethod
    def attempt_split_assignment(line_to_split: str) -> list or None:
        if '=' not in line_to_split:
            return None

        assignment_pos = line_to_split.find('=')
        before_assignment = line_to_split[0: assignment_pos].strip()
        after_assignment = line_to_split[assignment_pos + 1:].strip()

        if TruthScriptTokenizer.is_acceptable_id(before_assignment) is False:
            return None

        return [before_assignment, after_assignment]


class TruthScriptTokenizer:
    """
    TruthScriptTokenizer focuses on taking an expression as a string and breaking it down into
    a single Opo class that can be executed.
    """

    @staticmethod
    def tokenize(s: str, defined_vars: dict):

        # Start off with a list of string tokens and iteratively modify them to
        # be a list of Opo's  such that there is only 1 Opo left

        #  Part 1  Split Expression based on operators:s +-*/
        list_of_tokens_or_opos = TruthScriptTokenizer.tokenize_step1(s)

        #  Step 2 Merge Scientific Notation
        TruthScriptTokenizer.tokenize_step2_sci_notation(list_of_tokens_or_opos)

        #  Step 3 : start converting all the strings to Opo's
        TruthScriptTokenizer.tokenize_step3_str_conversion(list_of_tokens_or_opos, defined_vars)

        #  Step 4 take all binary operators and before/after tokens and make into a single OPT_BASE
        TruthScriptTokenizer.tokenize_step4_opo_conversion(list_of_tokens_or_opos)

        return list_of_tokens_or_opos[0]

    @staticmethod
    def tokenize_step1(s: str) -> list:
        list_of_items = []

        i = 0

        while True:
            next_splitter_index = TruthScriptTokenizer.step_1_helper_find_next_major_splitter(s, i, len(s))

            if next_splitter_index == len(s):
                sub = s[i: next_splitter_index]
                list_of_items.append(sub)
                break

            first_part = s[i: next_splitter_index].strip()
            token = s[next_splitter_index: next_splitter_index + 1].strip()

            list_of_items.append(first_part)
            list_of_items.append(token)

            i = next_splitter_index + 1
        return list_of_items

    @staticmethod
    def tokenize_step2_sci_notation(list_of_tokens_or_opos: list) -> None:
        for i in range(len(list_of_tokens_or_opos) - 1, -1, -1):
            string_token = str(list_of_tokens_or_opos[i])

            if string_token.endswith('e') and TruthScriptTokenizer.is_float(string_token[0: len(string_token) - 1]):

                next_str = str(list_of_tokens_or_opos[i + 1])

                is_next_str_float = TruthScriptTokenizer.is_float(next_str)
                is_next_str_positive_or_negative = next_str == '+' or next_str == '-'

                if is_next_str_float:

                    whole = string_token + next_str

                    list_of_tokens_or_opos.pop(i + 1)
                    list_of_tokens_or_opos[i] = whole

                elif is_next_str_positive_or_negative:
                    next_next_str = str(list_of_tokens_or_opos[i + 2])

                    if TruthScriptTokenizer.is_float(next_next_str):
                        whole = string_token + next_str + next_next_str
                        list_of_tokens_or_opos.pop(i + 2)
                        list_of_tokens_or_opos.pop(i + 1)
                        list_of_tokens_or_opos[i] = whole

    @staticmethod
    def tokenize_step3_str_conversion(list_of_tokens_or_opos: list, defined_vars: dict) -> None:
        for i in range(len(list_of_tokens_or_opos)):
            string_or_opo = list_of_tokens_or_opos[i]
            if not isinstance(string_or_opo, str):
                continue

            string_token = str(string_or_opo).strip()

            if string_token == '+' or string_token == '-' or string_token == '*' or string_token == '/' \
                    or string_token == '^':
                # ignore it since it will be dealt with in the next section
                pass
            elif i == 0 and string_token == '':
                next_str = str(list_of_tokens_or_opos[i + 1])
                if next_str == '-':
                    list_of_tokens_or_opos[0] = OpoConstantFloat(0.0)
                elif next_str == '+':
                    pass
            elif '(' in string_token:

                list_of_tokens_or_opos[i] = TruthScriptTokenizer.parse_function_or_grouping(string_token, defined_vars)
            elif string_token in defined_vars:
                list_of_tokens_or_opos[i] = OpoVariable(string_token)
            elif TruthScriptTokenizer.is_float(string_token):
                list_of_tokens_or_opos[i] = OpoConstantFloat(float(string_token))
            elif string_token.startswith("'") and string_token.endswith("'"):
                inner = string_token[1: len(string_token) - 1]
                list_of_tokens_or_opos[i] = OpoConstStr(inner)

    @staticmethod
    def tokenize_step4_opo_conversion(list_of_tokens_or_opos: list) -> None:
        while len(list_of_tokens_or_opos) != 1:
            precedence_index = TruthScriptTokenizer.step_4_helper_find_operator_index_based_on_precedence(
                list_of_tokens_or_opos)

            string_token = str(list_of_tokens_or_opos[precedence_index])
            bef = list_of_tokens_or_opos[precedence_index - 1]
            aft = list_of_tokens_or_opos[precedence_index + 1]

            list_of_tokens_or_opos.pop(precedence_index + 1)
            list_of_tokens_or_opos.pop(precedence_index)

            if string_token == '+':
                list_of_tokens_or_opos[precedence_index - 1] = OpoAdd(bef, aft)
            elif string_token == '-':
                list_of_tokens_or_opos[precedence_index - 1] = OpoSubtract(bef, aft)
            elif string_token == '*':
                list_of_tokens_or_opos[precedence_index - 1] = OpoMultiply(bef, aft)
            elif string_token == '/':
                list_of_tokens_or_opos[precedence_index - 1] = OpoDivide(bef, aft)
            elif string_token == '^':
                list_of_tokens_or_opos[precedence_index - 1] = OpoPower(bef, aft.const_float)

    @staticmethod
    def is_letter(s: str) -> bool:
        return ('a' <= s <= 'z') or ('A' <= s <= 'Z')

    @staticmethod
    def is_number(s: str) -> bool:
        return '0' <= s <= '9'

    @staticmethod
    def is_underscore(s: str) -> bool:
        return s == '_'

    @staticmethod
    def is_id_start_char(s: str) -> bool:
        return TruthScriptTokenizer.is_letter(s) or TruthScriptTokenizer.is_underscore(s)

    @staticmethod
    def is_id_char(s: str) -> bool:
        return TruthScriptTokenizer.is_letter(s) or TruthScriptTokenizer.is_number(
            s) or TruthScriptTokenizer.is_underscore(s)

    @staticmethod
    def is_acceptable_id(s: str) -> bool:
        if TruthScriptTokenizer.is_id_start_char(s[0]) is False:
            return False

        for i in range(1, len(s)):
            if TruthScriptTokenizer.is_id_char(s[i]) is False:
                return False
        return True

    @staticmethod
    def step_1_helper_find_next_major_splitter(s: str, start_index: int, end_index: int) -> int:
        level = 0

        is_in_str = False

        for i in range(start_index, end_index):
            c = s[i]

            if c == "'":
                if is_in_str:
                    is_in_str = False
                else:
                    is_in_str = True

            if is_in_str:
                continue

            if c == '(':
                level += 1
            if c == ')':
                level -= 1

            if level > 0:
                continue

            if c == '+' or c == '-' or c == '*' or c == '/' or c == '^':
                return i
        return end_index

    @staticmethod
    def find_function_param_split(s: str) -> list:
        split_parameters = []

        first_char_after_last_comma_index = 0

        level = 0

        for i in range(len(s)):
            c = s[i]

            if c == '(':
                level += 1
            if c == ')':
                level -= 1
            if level > 0:
                continue
            if c == ',':
                sub = s[first_char_after_last_comma_index: i]
                split_parameters.append(sub)
                first_char_after_last_comma_index = i + 1

        if first_char_after_last_comma_index < len(s):
            sub = s[first_char_after_last_comma_index:]
            split_parameters.append(sub)

        return split_parameters

    @staticmethod
    def parse_function_or_grouping(s: str, defined_vars: dict):
        open_paren_index = s.find('(')
        close_paren_index = s.rfind(')')

        before_parenthesis_part = s[0: open_paren_index].strip()
        between_parenthesis_part = s[open_paren_index + 1: close_paren_index].strip()
        after_parenthesis_part = s[close_paren_index + 1:].strip()

        if before_parenthesis_part == '':
            # if no identifier before the first parenthesis then we have a Grouping
            return TruthScriptTokenizer.tokenize(between_parenthesis_part, defined_vars)
        else:
            # if an identifier is before the first parenthesis,  we have a Function Call
            parameters = TruthScriptTokenizer.find_function_param_split(between_parenthesis_part)

            opo_list = []
            for parameter in parameters:
                opo_list.append(TruthScriptTokenizer.tokenize(parameter, defined_vars))
            return OpoFunctionCall(before_parenthesis_part, opo_list)

    @staticmethod
    def step_4_helper_find_operator_index_based_on_precedence(list_of_tokens_or_opos: list) -> int:
        for i in range(len(list_of_tokens_or_opos)):
            token_or_opo = list_of_tokens_or_opos[i]
            if isinstance(token_or_opo, str):
                if token_or_opo == '^':
                    return i

        for i in range(len(list_of_tokens_or_opos)):
            token_or_opo = list_of_tokens_or_opos[i]
            if isinstance(token_or_opo, str):
                if token_or_opo == '*' or token_or_opo == '/':
                    return i

        for i in range(len(list_of_tokens_or_opos)):
            token_or_opo = list_of_tokens_or_opos[i]
            if isinstance(token_or_opo, str):
                if token_or_opo == '+' or token_or_opo == '-':
                    return i

        return -1

    @staticmethod
    def is_float(s: str) -> bool:

        try:
            float(s)
            return True
        except ValueError:
            return False


class OpoConstantFloat:

    def __init__(self, const_float: float):
        self.const_float = const_float

    def execute_operation(self, special_stack: SpecialStack, calling_ktp: KnowledgeTensorPoint, var_lookup: dict,
                          source):
        return self.const_float

    def is_constant(self, c: float) -> bool:
        return self.const_float == c


class OpoConstStr:

    def __init__(self, s: str):
        self.constant_str = s

    def execute_operation(self, special_stack: SpecialStack, calling_ktp: KnowledgeTensorPoint, var_lookup: dict,
                          source):
        return self.constant_str


class OpoVariable:

    def __init__(self, v: str):
        self.var = v

    def execute_operation(self, special_stack: SpecialStack, calling_ktp: KnowledgeTensorPoint, var_lookup: dict,
                          source):
        return var_lookup[self.var]

    def is_constant(self, c: float) -> bool:
        return False


class OpoAdd:

    def __init__(self, left_opo, right_opo):
        self.left_opo = left_opo
        self.right_opo = right_opo

    @staticmethod
    def create(left_opo, right_opo):
        if left_opo.is_constant(0):
            return right_opo
        elif right_opo.is_constant(0):
            return left_opo
        else:
            return OpoAdd(left_opo, right_opo)

    def execute_operation(self, special_stack: SpecialStack, calling_ktp: KnowledgeTensorPoint, var_lookup: dict,
                          source):
        return Operations.addition_operation(
            self.left_opo.execute_operation(special_stack, calling_ktp, var_lookup, source),
            self.right_opo.execute_operation(special_stack, calling_ktp, var_lookup, source))

    def is_constant(self, c: float) -> bool:
        return False


class OpoSubtract:

    def __init__(self, left_opo, right_opo):
        self.left_opo = left_opo
        self.right_opo = right_opo

    @staticmethod
    def create(left_opo, right_opo):
        if right_opo.is_constant(0):
            return left_opo
        else:
            return OpoSubtract(left_opo, right_opo)

    def execute_operation(self, special_stack: SpecialStack, calling_ktp: KnowledgeTensorPoint, var_lookup: dict,
                          source):
        return Operations.subtraction_operation(
            self.left_opo.execute_operation(special_stack, calling_ktp, var_lookup, source),
            self.right_opo.execute_operation(special_stack, calling_ktp, var_lookup, source))

    def is_constant(self, c: float) -> bool:
        return False


class OpoMultiply:

    def __init__(self, left_opo, right_opo):
        self.left_opo = left_opo
        self.right_opo = right_opo

    @staticmethod
    def create(left_opo, right_opo):
        if left_opo.is_constant(0):
            return left_opo
        if right_opo.is_constant(0):
            return right_opo

        if left_opo.is_constant(1):
            return right_opo
        elif right_opo.is_constant(1):
            return left_opo
        else:
            return OpoMultiply(left_opo, right_opo)

    def execute_operation(self, special_stack: SpecialStack, calling_ktp: KnowledgeTensorPoint, var_lookup: dict,
                          source):
        return Operations.multiplication_operation(
            self.left_opo.execute_operation(special_stack, calling_ktp, var_lookup, source),
            self.right_opo.execute_operation(special_stack, calling_ktp, var_lookup, source))

    def is_constant(self, c: float) -> bool:
        return False


class OpoDivide:

    def __init__(self, left_opo, right_opo):
        self.left_opo = left_opo
        self.right_opo = right_opo

    @staticmethod
    def create(left_opo, right_opo):
        if right_opo.is_constant(1):
            return left_opo
        else:
            return OpoMultiply(left_opo, right_opo)

    def execute_operation(self, special_stack: SpecialStack, calling_ktp: KnowledgeTensorPoint, var_lookup: dict,
                          source):
        return Operations.division_operation(
            self.left_opo.execute_operation(special_stack, calling_ktp, var_lookup, source),
            self.right_opo.execute_operation(special_stack, calling_ktp, var_lookup, source))

    def is_constant(self, c: float) -> bool:
        return False


class OpoPower:

    def __init__(self, left_opo, right_float: float):
        self.left_opo = left_opo
        self.right_float = right_float

    @staticmethod
    def create(left_opo, right_float: float):
        if right_float == 1:
            return left_opo
        elif right_float == 0:
            return OpoConstantFloat(1)
        else:
            return OpoPower(left_opo, right_float)

    def execute_operation(self, special_stack: SpecialStack, calling_ktp: KnowledgeTensorPoint, var_lookup: dict,
                          source):
        return Operations.power_operation(
            self.left_opo.execute_operation(special_stack, calling_ktp, var_lookup, source), self.right_float)

    def is_constant(self, c: float) -> bool:
        return False


class OpoFunctionCall:

    def __init__(self, function_name: str, parameters: list):
        self.function_name = function_name
        self.parameters = parameters

    def execute_operation(self, special_stack: SpecialStack, calling_ktp: KnowledgeTensorPoint, var_lookup: dict,
                          source):
        param_values = []
        for i in range(len(self.parameters)):
            param_values.append(self.parameters[i].execute_operation(special_stack, calling_ktp, var_lookup, source))

        if self.function_name == 'get':
            parameters = []
            time_index = int(param_values[0])

            for i in range(1, len(param_values)):
                parameters.append(str(param_values[i]))

            if len(parameters) != 1:
                return None

            return special_stack.calculate(time_index, parameters[0])
        elif self.function_name == 'get_time':
            time_index = int(param_values[0])

            return special_stack.ut.time_range.index_int_to_float(time_index)

        elif self.function_name == 'get_age':
            time_index = int(param_values[0])

            time = special_stack.ut.time_range.index_int_to_float(time_index)
            age = time - special_stack.ut.user_filter.birthday
            return age
        elif self.function_name == 'uniform_dist':
            uniform_start = float(param_values[0])
            uniform_end = float(param_values[1])

            lower_bound = float(param_values[2])
            upper_bound = float(param_values[3])
            res = float(param_values[4])

            dist_to_return = ProbFloatDist.uniform_distribution(uniform_start, uniform_end, lower_bound, upper_bound,
                                                                res)

            return dist_to_return
        elif self.function_name == 'crash_error_core_tensor':
            raise Exception()


class SingleActionViewer(tk.Tk):
    """
    SingleActionViewer is a simple GUI that shows your current life score and the action that would improve it the most
    """

    def __init__(self, action_results: dict):
        super().__init__()

        self.title('SingleActionViewer')
        self.window_width = 987
        self.window_height = 654

        self.geometry(str(self.window_width + 20) + 'x' + str(self.window_height))

        self.main_canvas = tk.Canvas(self, width=self.window_width, height=self.window_height)
        self.main_canvas.pack()

        self.main_canvas.create_text(10, 10,
                                     text='This viewer is the SingleActionViewer. It is the simplest view possible of a UniversalKnowledgeTensor LifeScore computation.',
                                     anchor=tk.NW)
        self.main_canvas.create_text(10, 30,
                                     text='\tThe bare minimum information everyone wants to know is how they are doing in life and what can they do to make it better.',
                                     anchor=tk.NW)

        action_max = None
        action_max_result = action_results[Dimensions.action]

        for action in action_results.keys():
            if action_results[action] > action_max_result:
                action_max_result = action_results[action]
                action_max = action

        self.main_canvas.create_text(10, 70,
                                     text='Your life score is :' + format_float(action_results[Dimensions.action]),
                                     anchor=tk.NW)

        if action_max is None:
            self.main_canvas.create_text(10, 90, text='There is nothing you can do to increase your life score',
                                         anchor=tk.NW)
        else:
            self.main_canvas.create_text(10, 90,
                                         text='Performing ' + action_max + ' will increase your life score to '
                                              + format_float(action_max_result), anchor=tk.NW)


class MultiActionViewer(tk.Tk):
    """
    MultiActionViewer is a simple GUI that shows the resultant life scores for all the actions you can take.
    """

    def __init__(self, action_results: dict):
        super().__init__()

        self.title('MultiActionViewer')
        self.window_width = 987
        self.window_height = 654
        self.legend_width = 250
        self.value_label_width = 40
        self.bar_height = 30

        self.geometry(str(self.window_width + 20) + 'x' + str(self.window_height))

        self.main_canvas = tk.Canvas(self, width=self.window_width, height=self.window_height)
        self.main_canvas.pack()

        self.main_canvas.create_text(10, 10,
                                     text='This viewer is the MultiActionViewer. It provides an overview all LifeScores of every possible Action from a UniversalKnowledgeTensor LifeScore computation.',
                                     anchor=tk.NW)

        self.action_results = action_results
        self.draw_everything()

    def draw_everything(self) -> None:
        self.main_canvas.delete('tag')

        all_actions = list(self.action_results.keys())
        all_actions.sort(key=lambda action: -1 * self.action_results[action])

        max_value = self.action_results[all_actions[0]]
        min_value = self.action_results[all_actions[-1]]

        if min_value < 0.0 and max_value < 0.0:
            max_value = 0.0
        if min_value > 0.0 and max_value > 0.0:
            min_value = 0.0

        percentage_of_0 = (0.0 - min_value) / (max_value - min_value)
        x_pos_of_0 = int(self.legend_width + self.value_label_width + percentage_of_0 * (
                    self.window_width - self.legend_width - 2 * self.value_label_width))

        y_pos = self.bar_height * 3
        for action in all_actions:
            percentage_of_val = (self.action_results[action] - min_value) / (max_value - min_value)
            x_pos = int(self.legend_width + self.value_label_width + percentage_of_val * (
                        self.window_width - self.legend_width - 2 * self.value_label_width))

            action_text = action
            if action == Dimensions.reality:
                rect_fill = 'gray'
                action_text = 'reality'
            else:
                if self.action_results[action] < self.action_results[Dimensions.reality]:
                    rect_fill = 'red'
                else:
                    rect_fill = 'green'

            self.main_canvas.create_text(5, y_pos, text=action_text, anchor=tk.NW, tag='tag')
            self.main_canvas.create_rectangle(x_pos_of_0, y_pos, x_pos, y_pos + self.bar_height - 2, fill=rect_fill,
                                              tag='tag')

            if self.action_results[action] < 0:
                self.main_canvas.create_text(x_pos, y_pos, text=format_float(self.action_results[action]), anchor=tk.NE,
                                             tag='tag')
            else:
                self.main_canvas.create_text(x_pos, y_pos, text=format_float(self.action_results[action]), anchor=tk.NW,
                                             tag='tag')

            y_pos += self.bar_height


class RangeCoordMapper:
    """
    RangeCoordMapper is used by the CoreViewer to convert time_indexes into a x positions so that line charts can
    be drawn
    """

    def __init__(self, time_range: TimeRange):
        self.start_x_of_graph_lines = 100
        self.width_of_graph_content = -1
        self.width_of_graph_y_labels = 150

        self.min_index = 0
        self.max_index = time_range.num_time_epochs()

        self.time_range = time_range

    def index_to_position(self, index: int) -> float:
        percent = float(index - self.min_index) / float(self.max_index - self.min_index)

        if percent < 0.0:
            percent = 0.0
        if percent > 1.0:
            percent = 1.0

        pos_on_graph = percent * self.width_of_graph_content

        abs_pos = pos_on_graph + self.width_of_graph_y_labels + self.start_x_of_graph_lines

        return abs_pos


class GuiModes:
    """
    GuiModes is an enumeration that specifies the type of chart that can be displayed on the CoreViewer
    Currently, only line charts are supported but Stacked Area charts could also be displayed
    """
    name_only = 'name_only'
    line_chart_multi_action = 'line_chart_multi_action'


class TreeVisSettings:
    """
    The TreeVisSettings contains important setting information that is used by CoreViewerNodes in the CoreViewer GUI
    """

    def __init__(self, all_possible_actions: list):
        self.action_base = Dimensions.reality
        self.visible_actions = {}
        self.color_mapper = None
        self.coord_mapper = None

        self.color_mapper = ActionColorMapper(all_possible_actions)
        for action in all_possible_actions:
            self.visible_actions[action] = True


class ActionColorMapper:
    """
    The ActionColorMapper is used by the CoreViewer GUI to map actions to colors.
    """

    def __init__(self, all_actions: list):
        self.all_actions = all_actions
        self.action_to_color = {}

        self.distinct_colors = ['black', 'dark green', 'blue']

        for i in range(len(self.all_actions)):
            self.action_to_color[self.all_actions[i]] = self.distinct_colors[i % len(self.distinct_colors)]

        self.action_to_offset = {}

        for i in range(len(self.all_actions)):
            self.action_to_offset[self.all_actions[i]] = i

    def get_action_color(self, action: str) -> str:
        if action in self.action_to_color:
            return self.action_to_color[action]

        self.action_to_color[action] = self.distinct_colors[len(self.action_to_color) % len(self.distinct_colors)]

        return self.action_to_color[action]

    def get_action_offset(self, action: str) -> int:
        return self.action_to_offset[action]


class CoreViewerNode:
    """
    CoreViewerNode contains the logic for displaying every CoreData in the CoreViewer GUI.
    This class handles layout, drawing of the tree of inputs, and drawing the line chart
    """

    def __init__(self, parent, canvas_for_node, core_name: str, ktp_type: str, heavy_series: HeavySeries,
                 has_children: bool):
        self.gui_parent = parent
        self.canvas_for_node = canvas_for_node
        self.core_name = core_name
        self.formula_type = ktp_type
        self.heavy_series = heavy_series
        self.has_children = has_children

        self.mode = GuiModes.line_chart_multi_action
        self.gui_children = []

        self.left_abs_x_start = -1
        self.left_abs_x_end = -1
        self.left_abs_y_child_end = -1
        self.left_abs_y_end = -1
        self.left_abs_y_start = -1
        self.left_level = -1
        self.y_of_bottom_content = -1
        self.y_of_top_content = -1

        if self.gui_parent is not None:
            self.gui_parent.gui_children.append(self)

        self.line_color = 'black'
        self.selection_line_color = 'blue'
        self.selection_line_width = 3

        self.dash_config = (5, 3)
        self.dash_color = 'gray'

        self.left_is_children_expanded = False
        self.left_is_selected = False

        self.right_last_processed_start_x = -1

    def update_entire_tree(self, window_width: int) -> None:
        root_node = self.get_root()

        max_level = root_node.update_entire_tree_helper___left_compute_levels(0)

        max_level = int(((max_level / 5) + 1) * 5)

        root_node.update_entire_tree_helper___left_compute_y(0)

        start_x = CoreViewerNode.left_convert_level_to_x(max_level + 1)
        end_x = window_width
        root_node.update_entire_tree_helper___left_set_x_start_and_end(start_x, end_x)

    def update_entire_tree_helper___left_compute_levels(self, level: int) -> int:
        self.left_level = level
        max_child_level = level

        if self.left_is_children_expanded:
            for child in self.gui_children:
                m = child.update_entire_tree_helper___left_compute_levels(level + 1)
                max_child_level = max(max_child_level, m)

        return max_child_level

    def update_entire_tree_helper___left_compute_y(self, start_y: int) -> int:
        self.left_abs_y_start = start_y
        canvas_height = self.get_canvas_height()
        self.left_abs_y_end = start_y + canvas_height

        self.left_abs_y_child_end = self.left_abs_y_end

        if self.left_is_children_expanded:
            for child in self.gui_children:
                self.left_abs_y_child_end = child.update_entire_tree_helper___left_compute_y(self.left_abs_y_child_end)

        return self.left_abs_y_child_end

    def update_entire_tree_helper___left_set_x_start_and_end(self, start_x: int, end_x: int) -> None:
        self.left_abs_x_start = start_x
        self.left_abs_x_end = end_x

        if self.left_is_children_expanded:
            for child in self.gui_children:
                child.update_entire_tree_helper___left_set_x_start_and_end(start_x, end_x)

    def get_root(self):
        node = self
        while node.gui_parent is not None:
            node = node.gui_parent

        return node

    def set_mode(self, new_mode: str) -> None:
        self.mode = new_mode

    @staticmethod
    def get_line_chart_height() -> int:
        return 200

    def is_expand_collapse_action(self, x: int, y: int) -> str or None:
        x_of_level = CoreViewerNode.left_convert_level_to_x(self.left_level)

        if self.left_is_children_expanded:
            x_of_level = CoreViewerNode.left_convert_level_to_x(self.left_level + 1)

        y_of_center = self.get_control_center_y()

        if x_of_level - 5 <= x <= x_of_level + 5 and y_of_center - 5 <= y <= y_of_center + 5:
            return 'expand_collapse'
        else:
            return None

    def remove_all_children(self) -> None:

        for child in self.gui_children:
            child.remove_self_and_children()

        self.gui_children.clear()

    def remove_self_and_children(self) -> None:
        self.canvas_for_node.destroy()
        self.canvas_for_node = None

        for child in self.gui_children:
            child.remove_self_and_children()

        self.gui_children.clear()

    def set_canvas_positions(self) -> None:

        if self.canvas_for_node is not None:
            self.canvas_for_node.place(x=0, y=self.left_abs_y_start)

        for child in self.gui_children:
            child.set_canvas_positions()

    def get_child_core_viewer_node(self, child_core_name: str):

        for core_viewer in self.gui_children:
            if core_viewer.core_name == child_core_name:
                return core_viewer
        return None

    def get_canvas_height(self) -> int:
        if self.mode == GuiModes.name_only:
            return 25
        else:
            return CoreViewerNode.get_line_chart_height()

    @staticmethod
    def left_convert_level_to_x(level: int) -> int:
        return 10 + level * 20

    def get_control_center_y(self) -> int:
        return int(self.get_canvas_height() / 2)

    def draw_both_left_right(self, tvs: TreeVisSettings) -> None:

        if self.canvas_for_node is None:
            return

        self.draw_left(tvs)
        self.draw_right(tvs)

        if self.left_is_children_expanded:
            for child in self.gui_children:
                child.draw_both_left_right(tvs)

        self.canvas_for_node.delete('selection')

        # put a border around it if it is selected
        if self.left_is_selected:
            # top line
            self.canvas_for_node.create_line(self.left_abs_x_start, 5, self.left_abs_x_end, 5,
                                             fill=self.selection_line_color, width=self.selection_line_width,
                                             tag='selection')
            # bottom line
            self.canvas_for_node.create_line(self.left_abs_x_start, self.get_canvas_height() - 3, self.left_abs_x_end,
                                             self.get_canvas_height() - 3, fill=self.selection_line_color,
                                             width=self.selection_line_width, tag='selection')
            # left
            self.canvas_for_node.create_line(self.left_abs_x_start, 5, self.left_abs_x_start,
                                             self.get_canvas_height() - 3, fill=self.selection_line_color,
                                             width=self.selection_line_width, tag='selection')
            # right
            self.canvas_for_node.create_line(self.left_abs_x_end - 2, 5, self.left_abs_x_end - 2,
                                             self.get_canvas_height() - 3, fill=self.selection_line_color,
                                             width=self.selection_line_width, tag='selection')

    def draw_left(self, tvs: TreeVisSettings) -> None:
        x_of_level = CoreViewerNode.left_convert_level_to_x(self.left_level)
        center_y = self.get_control_center_y()

        self.canvas_for_node.config(height=self.get_canvas_height())
        self.canvas_for_node.delete('left')

        # horizontal line
        self.canvas_for_node.create_line(x_of_level, center_y, self.left_abs_x_start, center_y, fill=self.line_color,
                                         tag='left')

        if self.gui_parent is not None:
            child_index = self.gui_parent.gui_children.index(self)

            # Vertical Line Below Center
            if child_index == 0:
                self.gui_parent.draw_left_helper_vertical_center_to_bottom(x_of_level)
            else:
                prev_sibling = self.gui_parent.gui_children[child_index - 1]
                prev_sibling.draw_left_helper_vertical_center_to_bottom(x_of_level)

            self.draw_left_helper_vertical_top_to_center(x_of_level)

        # Draw Expansion Box
        if self.has_children and self.left_is_children_expanded is False:
            self.canvas_for_node.create_line(x_of_level - 5, center_y + 5, x_of_level + 5, center_y + 5,
                                             fill=self.line_color, tag='left')
            self.canvas_for_node.create_line(x_of_level - 5, center_y - 5, x_of_level + 5, center_y - 5,
                                             fill=self.line_color, tag='left')

            self.canvas_for_node.create_line(x_of_level - 5, center_y - 5, x_of_level - 5, center_y + 5,
                                             fill=self.line_color, tag='left')
            self.canvas_for_node.create_line(x_of_level + 5, center_y - 5, x_of_level + 5, center_y + 5,
                                             fill=self.line_color, tag='left')

            self.canvas_for_node.create_line(x_of_level - 5, center_y, x_of_level, center_y, fill=self.line_color,
                                             tag='left')
            self.canvas_for_node.create_line(x_of_level, center_y - 5, x_of_level, center_y + 5, fill=self.line_color,
                                             tag='left')

        # display the type
        if self.formula_type == KtpTypeDef.aggr:
            self.canvas_for_node.create_text(self.left_abs_x_start - 15, center_y - 20, text='Aggregate', anchor=tk.E,
                                             tag='left')
        elif self.formula_type == KtpTypeDef.dir_est:
            self.canvas_for_node.create_text(self.left_abs_x_start - 15, center_y - 20, text='Estimate', anchor=tk.E,
                                             tag='left')
        elif self.formula_type == KtpTypeDef.dir_meas:
            self.canvas_for_node.create_text(self.left_abs_x_start - 15, center_y - 20, text='Measurement', anchor=tk.E,
                                             tag='left')
        elif self.formula_type == KtpTypeDef.life_score_event:
            self.canvas_for_node.create_text(self.left_abs_x_start - 15, center_y - 20, text='Event', anchor=tk.E,
                                             tag='left')
        elif self.formula_type == KtpTypeDef.life_score_state:
            self.canvas_for_node.create_text(self.left_abs_x_start - 15, center_y - 20, text='State', anchor=tk.E,
                                             tag='left')

    def draw_left_helper_vertical_top_to_bottom(self, x_pos: int) -> None:
        self.canvas_for_node.create_line(x_pos, 0, x_pos, self.get_canvas_height(), fill=self.line_color, tag='left')

        for child in self.gui_children:
            child.draw_left_helper_vertical_top_to_bottom(x_pos)

    def draw_left_helper_vertical_top_to_center(self, x_pos: int) -> None:
        self.canvas_for_node.create_line(x_pos, 0, x_pos, self.get_canvas_height() / 2, fill=self.line_color,
                                         tag='left')

    def draw_left_helper_vertical_center_to_bottom(self, x_pos: int) -> None:
        canvas_height = self.get_canvas_height()

        if self.canvas_for_node is not None:
            self.canvas_for_node.create_line(x_pos, canvas_height / 2, x_pos, canvas_height, fill=self.line_color,
                                             tag='left')

        for child in self.gui_children:
            child.draw_left_helper_vertical_top_to_bottom(x_pos)

    def draw_right(self, tvs: TreeVisSettings) -> None:
        if self.left_abs_x_start == self.right_last_processed_start_x:
            return

        self.right_last_processed_start_x = self.left_abs_x_start

        top_offset = 30
        bottom_offset = 10

        s = Name.create_english_description(self.core_name)

        new_start_y = 0 + top_offset
        new_end_y = self.get_canvas_height() - bottom_offset

        self.canvas_for_node.delete('right')
        self.canvas_for_node.create_text(self.left_abs_x_start + 25, 5, text=s, anchor=tk.NW, tag='right')

        if self.mode == GuiModes.name_only:
            pass
        elif self.mode == GuiModes.line_chart_multi_action:
            self.draw_right_line_chart(self.left_abs_x_start, new_start_y, self.left_abs_x_end, new_end_y, tvs)

    def calculate_min_max_array(self, tvs: TreeVisSettings):

        min_maxs = MinMaxList()

        for i in range(tvs.coord_mapper.min_index, tvs.coord_mapper.max_index):
            min_maxs.append_new_min_maxer()

            core_data_at_time = self.heavy_series.list_of_core_data_at_times[i]
            values = core_data_at_time.retrieve_action_to_values_float_for_gui()

            for value in values.keys():
                if value == tvs.action_base or value in tvs.visible_actions:
                    val = values[value]
                    if isinstance(val, float):
                        min_maxs.incorporate(value, val)
                    elif isinstance(val, ProbFloatDist):
                        min_maxs.incorporate(value, val.calculate_mean())
        return min_maxs

    def draw_right_line_chart(self, start_x: int, start_y: int, end_x: int, end_y: int, tvs: TreeVisSettings) -> None:

        min_max_array = self.calculate_min_max_array(tvs)

        min_val = min_max_array.min_max_over_all_items.sorted_item_float_list[0].f
        max_val = min_max_array.min_max_over_all_items.sorted_item_float_list[-1].f

        img_height = end_y - start_y

        self.draw_right_line_chart_helper___y_label_horizontal_lines(start_x, start_y, img_height, min_val, max_val,
                                                                     tvs)

        dict_of_values = self.draw_right_line_chart_helper___content_vert_lines(start_y, end_y, min_val, max_val, tvs)

        distances = min_max_array.determine_farthest_off_from_second(dict_of_values.keys())
        self.draw_right_line_chart_helper___action_labels(distances, tvs, min_val, max_val)

    def draw_right_line_chart_helper___y_label_horizontal_lines(self, start_x: int, start_y: int, img_height: int,
                                                                min_val: float, max_val: float,
                                                                tvs: TreeVisSettings) -> None:

        horizontal_lane_height = 40
        num_hor_lines = int(math.floor(img_height / horizontal_lane_height)) + 1

        val_range = max_val - min_val
        val_range_per_lane = val_range / (num_hor_lines - 1)

        height_per_lane = img_height / num_hor_lines

        width_of_the_description = 100

        self.y_of_top_content = -1
        self.y_of_bottom_content = -1

        for i in range(num_hor_lines):
            val_to_display = min_val + i * val_range_per_lane
            y_bottom = start_y + img_height - i * height_per_lane
            y_center = y_bottom - height_per_lane / 2
            y_top = y_bottom - height_per_lane

            str_to_display = format_float(val_to_display)
            self.canvas_for_node.create_text(start_x + width_of_the_description - 10, y_top, text=str_to_display,
                                             anchor=tk.E, tag='right')

            x_left = int(tvs.coord_mapper.index_to_position(tvs.coord_mapper.min_index))
            x_right = int(tvs.coord_mapper.index_to_position(tvs.coord_mapper.max_index - 1))
            self.canvas_for_node.create_line(x_left, y_center, x_right, y_center, dash=self.dash_config,
                                             fill=self.dash_color, tag='right')

            if i == 0:
                self.y_of_bottom_content = y_center
            if i == num_hor_lines - 1:
                self.y_of_top_content = y_center

    def draw_right_line_chart_helper___content_vert_lines(self, start_y: int, end_y: int, min_val: float,
                                                          max_val: float, tvs: TreeVisSettings) -> dict:
        prev_x_pos = 0

        prev_dict_values = {}

        for time_index in range(tvs.coord_mapper.min_index, tvs.coord_mapper.max_index):
            x_pos = int(tvs.coord_mapper.index_to_position(time_index))

            sub_heavy_series_at_t = self.heavy_series.list_of_core_data_at_times[time_index]
            dict_of_values = sub_heavy_series_at_t.retrieve_action_to_values_float_for_gui()

            self.canvas_for_node.create_line(int(x_pos), start_y, int(x_pos), end_y, dash=self.dash_config,
                                             fill=self.dash_color, tag='right')

            # apply a small offset to each action so that 2 actions with the same value are not hidden
            action_offset = 0

            for action in dict_of_values.keys():
                val = dict_of_values[action]

                action_color = tvs.color_mapper.get_action_color(action)

                if isinstance(val, float):
                    percent = (val - min_val) / (max_val - min_val)
                    y_pos = action_offset + int(
                        (self.y_of_top_content - self.y_of_bottom_content) * percent + self.y_of_bottom_content)

                    self.canvas_for_node.create_oval(x_pos - 1, y_pos - 1, x_pos + 1, y_pos + 1, fill=action_color,
                                                     tag='right')

                if isinstance(val, float) and action in prev_dict_values and isinstance(prev_dict_values[action],
                                                                                        float):
                    percent1 = (val - min_val) / (max_val - min_val)
                    curr_y_pos = action_offset + int(
                        (self.y_of_top_content - self.y_of_bottom_content) * percent1 + self.y_of_bottom_content)

                    percent2 = (prev_dict_values[action] - min_val) / (max_val - min_val)
                    prev_y_pos = action_offset + int(
                        (self.y_of_top_content - self.y_of_bottom_content) * percent2 + self.y_of_bottom_content)

                    self.canvas_for_node.create_line(prev_x_pos, prev_y_pos, x_pos, curr_y_pos, fill=action_color,
                                                     tag='right', width=2)

                # action_offset disabled for now. Consider adding it when there are too many actions close to each other
                action_offset += 0

            prev_dict_values = dict_of_values

            prev_x_pos = x_pos

        return prev_dict_values

    def draw_right_line_chart_helper___action_labels(self, distances: dict, tvs: TreeVisSettings, min_val: float,
                                                     max_val: float) -> None:
        val_range = max_val - min_val

        missing_actions = []
        for action in distances.keys():
            # placing labels in the middle of the graph instead of the legend needs some work
            if True:
                missing_actions.append(action)
                continue

            min_max = distances[action]
            if len(min_max.sorted_item_float_list) == 0:
                missing_actions.append(action)
                continue

            highest = min_max.sorted_item_float_list[-1]
            highest_item = highest.item
            time_index = int(highest_item[0])
            val = highest_item[1]
            distance = highest.f

            if abs(distance) < 0.05 * val_range:
                missing_actions.append(action)
                continue

            if time_index == tvs.coord_mapper.max_index - 1:
                missing_actions.append(action)
                continue

            x_pos = tvs.coord_mapper.index_to_position(time_index)

            percent = (val - min_val) / (max_val - min_val)
            y_pos = (self.y_of_top_content - self.y_of_bottom_content) * percent + self.y_of_bottom_content

            if distance > 0:
                y_pos -= 10
            elif distance < 0:
                y_pos += 10

            action_text = action.split(MultiDimLoc.hierarchy)[-1]
            if action == Dimensions.reality:
                action_text = 'reality'
            action_text = action_text.replace('_', ' ')

            self.canvas_for_node.create_text(int(x_pos), int(y_pos), text=action_text,
                                             fill=tvs.color_mapper.get_action_color(action), tag='right')

        missing_actions.sort()

        x_of_legend = int(tvs.coord_mapper.index_to_position(tvs.coord_mapper.max_index - 1) + 10)

        for i in range(len(missing_actions)):
            missing_action = missing_actions[i]
            y_of_legend = int(self.y_of_top_content + i * 20 - 10)
            if y_of_legend > self.y_of_bottom_content + 20:
                break

            action_text = missing_action.split(MultiDimLoc.hierarchy)[-1]
            if missing_action == Dimensions.reality:
                action_text = 'reality'
            action_text = action_text.replace('_', ' ')

            self.canvas_for_node.create_text(x_of_legend, y_of_legend, text=action_text,
                                             fill=tvs.color_mapper.get_action_color(missing_action), tag='right',
                                             anchor=tk.NW)


class CoreViewer(tk.Tk):
    """
    CoreViewer is a gui that shows a line chart for every CoreData across time. It also can be expanded to show the
    input CoreData's
    """

    def __init__(self, ut: UniversalKnowledgeTensor, range_coord_mapper: RangeCoordMapper):
        super().__init__()

        self.title('CoreViewer')
        self.window_width = 987
        self.window_height = 654

        self.top_canvas_height = 40
        self.bottom_canvas_height = 40

        self.last_img_height = 0

        self.geometry(str(self.window_width + 20) + 'x' + str(self.window_height))

        self.top_frame = tk.Frame(self)
        self.top_frame.pack(side=tk.TOP, anchor=tk.NW)  # , fill=tk.X
        self.top_canvas = tk.Canvas(self.top_frame, width=self.window_width, height=self.top_canvas_height)
        self.top_canvas.pack(side=tk.TOP, fill=tk.X)

        self.bottom_canvas_scroll = tk.Canvas(self)
        self.bottom_frame = tk.Frame(self.bottom_canvas_scroll, bd=0, width=self.window_width)
        self.bottom_scroll_bar = tk.Scrollbar(self.bottom_canvas_scroll, orient='vertical',
                                              command=self.bottom_canvas_scroll.yview)
        self.bottom_canvas_scroll.configure(yscrollcommand=self.bottom_scroll_bar.set)
        self.bottom_canvas_scroll.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        self.bottom_scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.bottom_window = self.bottom_canvas_scroll.create_window((0, 0), window=self.bottom_frame, anchor='n')

        self.bind('<Configure>', self.on_frame_configure)

        self.ut = ut
        self.ut_starting_ktp = self.ut.last_starting_point
        self.range_coord_mapper = range_coord_mapper

        all_actions = ut.retrieve_all_actions()

        self.tvs = TreeVisSettings(all_actions)
        self.tvs.coord_mapper = range_coord_mapper

        ft = ut.series_dependence_tracker.compute_max_formula_type(self.ut_starting_ktp)

        c = tk.Canvas(self.bottom_frame, width=self.window_width, height=CoreViewerNode.get_line_chart_height())
        c.bind('<Button-1>', self.canvas_click)

        has_children = len(ut.series_dependence_tracker.inputs_of_core_name[self.ut_starting_ktp]) > 0
        self.root_viewer_node = CoreViewerNode(None, c, self.ut_starting_ktp, ft,
                                               ut.retrieve_heavy_series(self.ut_starting_ktp), has_children)
        c.core_viewer_node = self.root_viewer_node

        self.selected_viewer_node = self.root_viewer_node
        self.expand_selected_node()

        self.update_all_displays()

    def on_frame_configure(self, event=None):
        self.bottom_canvas_scroll.configure(scrollregion=self.bottom_canvas_scroll.bbox('all'))

    def update_all_displays(self) -> None:
        self.root_viewer_node.update_entire_tree(self.window_width)
        self.root_viewer_node.set_canvas_positions()

        self.range_coord_mapper.start_x_of_graph_lines = self.root_viewer_node.left_abs_x_start
        self.range_coord_mapper.width_of_graph_y_labels = 100
        self.range_coord_mapper.width_of_graph_content = self.window_width - self.range_coord_mapper.start_x_of_graph_lines - self.range_coord_mapper.width_of_graph_y_labels - 120
        img_height = self.root_viewer_node.left_abs_y_child_end

        self.draw_header()

        self.root_viewer_node.draw_both_left_right(self.tvs)

        self.bottom_frame.config(height=img_height)
        self.last_img_height = img_height

    def canvas_click(self, evt) -> None:
        selected_canvas = evt.widget

        if self.selected_viewer_node is not None:
            self.selected_viewer_node.left_is_selected = False

        self.selected_viewer_node = selected_canvas.core_viewer_node
        self.selected_viewer_node.left_is_selected = True

        action = self.selected_viewer_node.is_expand_collapse_action(evt.x, evt.y)
        if action == 'expand_collapse':
            self.expand_selected_node()

        self.update_all_displays()

    def expand_selected_node(self) -> None:
        if self.selected_viewer_node is None:
            msg.showinfo('TruthCoin', 'Please select a node')
            return
        if self.selected_viewer_node.core_name not in self.ut.series_dependence_tracker.inputs_of_core_name:
            msg.showinfo('TruthCoin', 'No Children in node' + self.selected_viewer_node.core_name)
            return

        if self.selected_viewer_node.left_is_children_expanded is False:

            self.selected_viewer_node.gui_children.clear()
            self.selected_viewer_node.left_is_children_expanded = True
            inputs = self.ut.series_dependence_tracker.inputs_of_core_name[self.selected_viewer_node.core_name]

            for input_child in inputs.keys():
                has_children = input_child in self.ut.series_dependence_tracker.inputs_of_core_name and len(
                    self.ut.series_dependence_tracker.inputs_of_core_name[input_child]) > 0

                ft = self.ut.series_dependence_tracker.compute_max_formula_type(input_child)

                c = tk.Canvas(self.bottom_frame, width=self.window_width, height=CoreViewerNode.get_line_chart_height())
                c.bind('<Button-1>', self.canvas_click)
                core_viewer_node = CoreViewerNode(self.selected_viewer_node, c, input_child, ft,
                                                  self.ut.retrieve_heavy_series(input_child), has_children)
                c.core_viewer_node = core_viewer_node

                core_viewer_node.set_mode(self.selected_viewer_node.mode)
        else:
            self.selected_viewer_node.remove_all_children()

            self.selected_viewer_node.left_is_children_expanded = False

        self.update_all_displays()

    def draw_header(self) -> None:
        self.top_canvas.delete('header')

        y_pos = 10
        for i in range(self.tvs.coord_mapper.min_index, self.tvs.coord_mapper.max_index):
            time = self.tvs.coord_mapper.time_range.index_int_to_float(i)

            x_pos = int(self.tvs.coord_mapper.index_to_position(i))
            s = str(int(time))

            self.top_canvas.create_text(x_pos, y_pos, text=s, fill='black', anchor=tk.CENTER, tag='header')
            self.top_canvas.create_line(x_pos, y_pos + 20, x_pos, self.top_canvas_height, tag='header')

    def expand_to_specified_core_name(self, specified_core_name: str) -> None:

        parent_chain = self.ut.series_dependence_tracker.get_parent_chain(specified_core_name)

        parent_chain.pop(0)
        self.selected_viewer_node = self.root_viewer_node

        for child_core_name in parent_chain:
            self.expand_selected_node()

            self.selected_viewer_node = self.selected_viewer_node.get_child_core_viewer_node(child_core_name)

        self.selected_viewer_node.left_is_selected = True


class ItemFloat:
    """
    ItemFloat is a simple pairing of an item and a float. It is used by the MinMaxer
    to keep track of which item had which value.
    """

    def __init__(self, item, f: float):
        self.item = item
        self.f = f


class MinMaxer:
    """
    MinMaxer will take a list of ItemFloats and keep track of which 2 have the lowest value
    and which 2 have the highest value. This is used by the CoreViewer to label some of the lines
    """

    def __init__(self):
        self.sorted_item_float_list = []

    def incorporate(self, item, val: float) -> None:
        self.sorted_item_float_list.append(ItemFloat(item, val))
        self.sorted_item_float_list.sort(key=lambda item_float: item_float.f)

        if len(self.sorted_item_float_list) == 5:
            self.sorted_item_float_list.pop(2)

    def incorporate_abs(self, item, val: float) -> None:
        self.sorted_item_float_list.append(ItemFloat(item, val))
        self.sorted_item_float_list.sort(key=lambda item_float: abs(item_float.f))

        if len(self.sorted_item_float_list) == 5:
            self.sorted_item_float_list.pop(2)


class MinMaxList:
    """
    MinMaxList is a list of MinMaxer's. Its main use is in the CoreViewer to determine
    which action is farthest from the others so it can be labeled.
    """

    def __init__(self):
        self.min_max_over_all_items = MinMaxer()
        self.min_max_list = []

    def append_new_min_maxer(self) -> None:
        self.min_max_list.append(MinMaxer())

    def incorporate(self, item, val: float) -> None:
        self.min_max_over_all_items.incorporate(item, val)
        self.min_max_list[-1].incorporate(item, val)

    def determine_farthest_off_from_second(self, keys) -> dict:
        farthest_off_from_second = {}
        for key in keys:
            farthest_off_from_second[key] = MinMaxer()

        for time_index in range(len(self.min_max_list)):

            min_maxer_at_time = self.min_max_list[time_index]

            if len(min_maxer_at_time.sorted_item_float_list) == 1:
                continue

            lower_distance = min_maxer_at_time.sorted_item_float_list[0].f - min_maxer_at_time.sorted_item_float_list[
                1].f
            lower_object = str(min_maxer_at_time.sorted_item_float_list[0].item)
            lower_value = min_maxer_at_time.sorted_item_float_list[0].f

            farthest_off_from_second[lower_object].incorporate_abs([time_index, lower_value], lower_distance)

            upper_distance = min_maxer_at_time.sorted_item_float_list[-1].f - min_maxer_at_time.sorted_item_float_list[
                -2].f
            upper_object = str(min_maxer_at_time.sorted_item_float_list[-1].item)
            upper_value = min_maxer_at_time.sorted_item_float_list[-1].f

            farthest_off_from_second[upper_object].incorporate_abs([time_index, upper_value], upper_distance)
        return farthest_off_from_second


def format_float(val: float) -> str:
    abs_val = abs(val)

    if 0.1 <= abs_val <= 1000000:
        return '%.5f' % val
    else:
        return '%.8e' % val


class SecureHash:
    """
    SecureHash is a temporary placeholder for a hashing algorithm. KnowledgeTensors and KnowledgeTensorPoints are
    identified by their hash
    """

    def __init__(self):
        self.hsh = 0
        self.input_index = 0

    @staticmethod
    def get_hash_8bit(i: int) -> int:
        shift_amount = i % 24
        shift_amount_2 = (i + 7) % 24

        flip = 0xFF - i

        return (flip << shift_amount) + (i << shift_amount_2)

    def update(self, s: str) -> None:
        for character in s:
            x = ord(character)
            h = SecureHash.get_hash_8bit(x)
            self.hsh = self.hsh ^ h ^ (self.input_index << 13)
            self.input_index += 1

    def digest(self) -> int:
        return self.hsh


class JsonTreeNode:
    """
    JsonTreeNode is used by JsonLogger to capture all the information for a single Node
    """

    def __init__(self, parent):
        self.parent = parent

        self.small_children_keys = []
        self.small_children_dict = {}
        self.large_children_list = []
        self.large_children_name = None

    def create_child_node(self):
        child_node = JsonTreeNode(self)
        self.large_children_list.append(child_node)
        return child_node

    def set_small_child(self, child_key: str, item) -> None:
        self.small_children_dict[child_key] = item
        self.small_children_keys.append(child_key)

    @staticmethod
    def format_string(value: str) -> str:
        return '"' + value.replace('\t', '    ') + '"'

    @staticmethod
    def format_value(value) -> str:
        if isinstance(value, str):
            return JsonTreeNode.format_string(value)
        elif isinstance(value, float):
            return format_float(value)
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, ProbFloatDist):
            return JsonTreeNode.format_string('mean=' + format_float(value.calculate_mean()))

    def log_to_json(self, file, spacing: str, level: int, comma_at_end: bool) -> None:

        total_spacing = spacing * level
        if len(self.small_children_keys) == 0:
            file.write(total_spacing + '{}\n')
            return

        first_key = self.small_children_keys[0]

        file.write(total_spacing + '{' + JsonTreeNode.format_string(first_key) + ': ' + JsonTreeNode.format_value(
            self.small_children_dict[first_key]))

        for i in range(1, len(self.small_children_keys)):
            key = self.small_children_keys[i]
            file.write(', ' + JsonTreeNode.format_string(key) + ': ' + JsonTreeNode.format_value(
                self.small_children_dict[key]))

        str_comma_at_end = ''
        if comma_at_end:
            str_comma_at_end = ','

        if len(self.large_children_list) == 0:
            file.write('}' + str_comma_at_end + '\n')
        else:
            file.write(', ' + JsonTreeNode.format_string(self.large_children_name) + ': [\n')

            for i in range(len(self.large_children_list)):
                show_comma_at_end = i < len(self.large_children_list) - 1
                if isinstance(self.large_children_list[i], JsonTreeNode):
                    json_node = self.large_children_list[i]
                    json_node.log_to_json(file, spacing, level + 1, show_comma_at_end)
                else:
                    file.write(total_spacing + spacing + JsonTreeNode.format_value(self.large_children_list[i]))
                    if show_comma_at_end:
                        file.write(',\n')
                    else:
                        file.write('\n')
            file.write(total_spacing + ']}' + str_comma_at_end + '\n')


class JsonLogger:
    """
    JsonLogger is used to log a tree of information into a json file
    """

    def __init__(self):
        self.active_node = JsonTreeNode(None)

    def create_node(self) -> JsonTreeNode:
        child_node = self.active_node.create_child_node()
        self.active_node = child_node
        return child_node

    def go_to_parent(self) -> None:
        self.active_node = self.active_node.parent

    def log_to_json(self, file_name: str) -> None:
        file = open(file_name, 'w')

        self.active_node.log_to_json(file, '    ', 0, False)

        file.close()

    def set_small_child(self, child_key: str, item) -> None:
        self.active_node.set_small_child(child_key, item)


class TestCurve:
    """
    TestCurve is a collection of 3 y points on the cartesian coordinate plane and they are used to represent
    a curve. This is a test generation class and should be removed when there are experts who have
    encoded their knowledge into KnowledgeTensors.
    """

    def __init__(self, y_start: float, y_mid: float, y_end: float):
        self.y_start = y_start
        self.y_mid = y_mid
        self.y_end = y_end

    def __add__(self, other):
        if isinstance(other, float):
            return TestCurve(self.y_start + other, self.y_mid + other, self.y_end + other)
        elif isinstance(other, TestCurve):
            return TestCurve(self.y_start + other.y_start, self.y_mid + other.y_mid, self.y_end + other.y_end)

    def __truediv__(self, other: float):
        return TestCurve(self.y_start / other, self.y_mid / other, self.y_end / other)

    def __mul__(self, other):
        return TestCurve(self.y_start * other, self.y_mid * other, self.y_end * other)

    @staticmethod
    def get_y_percent(y: float, y_start: float, y_end: float) -> float:
        if y_end == y_start:
            return 1.0

        return (y - y_start) / (y_end - y_start)

    def split_n_py_percent(self, n_py_start: float, n_py_percent_growth_share: float) -> list:

        n_py_end = (self.y_end / self.y_start) * n_py_percent_growth_share * n_py_start
        n_py_mid = (self.y_mid / self.y_start) * n_py_percent_growth_share * n_py_start

        other_start = self.y_start / n_py_start
        other_end = self.y_start / (n_py_percent_growth_share * n_py_start)
        other_mid = self.y_mid / n_py_mid

        return [TestCurve(n_py_start, n_py_mid, n_py_end), TestCurve(other_start, other_mid, other_end)]

    def split_add_with_converter(self, converters: dict, percent_starts: dict, percent_ends: dict) -> dict:
        missing_starts = []
        missing_growths = []
        total_specified_growths = 0.0
        total_specified_starts = 0.0
        for key in converters.keys():
            if key not in percent_starts:
                missing_starts.append(key)
            else:
                total_specified_starts += percent_starts[key]

            if key not in percent_ends:
                missing_growths.append(key)
            else:
                total_specified_growths += percent_ends[key]

        if len(missing_starts) != 0:
            fill_in_missing_start = (1.0 - total_specified_starts) / len(missing_starts)
            for key in converters.keys():
                if key not in percent_starts:
                    percent_starts[key] = fill_in_missing_start

        if len(missing_growths) != 0:
            fill_in_missing_growth = (1.0 - total_specified_growths) / len(missing_growths)
            for key in converters.keys():
                if key not in percent_ends:
                    percent_ends[key] = fill_in_missing_growth

        split_y3s = {}
        for key in converters.keys():
            raw_start = self.y_start * percent_starts[key]
            percent_mid = (percent_starts[key] + percent_ends[key]) / 2.0
            raw_mid = self.y_mid * percent_mid
            raw_end = self.y_end * percent_ends[key]

            raw = TestCurve(raw_start, raw_mid, raw_end)
            adjusted = raw / converters[key]
            split_y3s[key] = adjusted

        return split_y3s

    def calculate_lower_val(self) -> float:
        return min(self.y_start, min(self.y_mid, self.y_end))

    def calculate_upper_val(self) -> float:
        return max(self.y_start, max(self.y_mid, self.y_end))

    def interpolate_left(self, time_percent: float) -> float:
        return (1 - time_percent) * self.y_start + time_percent * self.y_mid

    def interpolate_right(self, time_percent: float) -> float:
        return (1 - time_percent) * self.y_mid + time_percent * self.y_end

    def interpolate(self, time_percent: float) -> float:
        if time_percent < 0.5:
            return self.interpolate_left(time_percent * 2.0)
        else:
            return self.interpolate_right(time_percent * 2.0 - 1.0)

    @staticmethod
    def vary_single(val: float, percent: float) -> float:
        low_val = val * 0.999
        high_val = val * 1.001

        new_val = percent * (high_val - low_val) + low_val

        return new_val

    def create_variation(self, start_percent: float, mid_percent: float, end_percent: float):
        return TestCurve(TestCurve.vary_single(self.y_start, start_percent),
                         TestCurve.vary_single(self.y_mid, mid_percent), TestCurve.vary_single(self.y_end, end_percent))

    def create_quadratic(self, start_time: float, end_time: float):
        return TestQuadFunc.construct_3_points(start_time, self.y_start, (start_time + end_time) / 2, self.y_mid,
                                               end_time, self.y_end)

    def create_segment(self, start_time: float, left_percent: float, end_time: float, right_percent: float):
        return TestSegment(start_time, TestCurve.vary_single(self.y_start, left_percent), end_time,
                           TestCurve.vary_single(self.y_end, right_percent))


class TestLinearFunc:
    """
    TestLinearFunc is a simple representation of a linear function with slope and offset.
    It is used in the Test KnowledgeTensorPoint generation to simulate knowledge as a formula. This is a test
        generation class and should be removed when there are experts who have encoded their knowledge into
        KnowledgeTensors.
    """

    def __init__(self):
        self.slope = 0.0
        self.offset = 0.0

    @staticmethod
    def construct_offset_slope(offset: float, slope: float):
        linear_func = TestLinearFunc()
        linear_func.offset = offset
        linear_func.slope = slope
        return linear_func

    def create_formatted_str(self, var_name: str) -> str:
        formatted_polynomial = ''
        coefficients = self.generate_coefficients()

        for i in range(len(coefficients)):
            single_coefficient = coefficients[i]

            if single_coefficient == 0:
                continue

            positive_coefficient = abs(single_coefficient)
            if single_coefficient >= 0:
                if i != 0:
                    formatted_polynomial += '+ '
            else:
                formatted_polynomial += '- '

            if i > 0 and positive_coefficient == 1:
                # don't show number
                pass
            elif positive_coefficient > 0.01:

                format_template = '%.5f'

                formatted_polynomial += format_template % positive_coefficient
            else:

                format_template = '%.5e'
                formatted_polynomial += format_template % positive_coefficient

            if i == 1:
                if positive_coefficient != 1.0:
                    formatted_polynomial += '*'
                formatted_polynomial += var_name
            elif i >= 2:
                if positive_coefficient != 1.0:
                    formatted_polynomial += '*'
                formatted_polynomial += var_name + '^' + str(i)

            formatted_polynomial += ' '

        return formatted_polynomial

    def eval(self, x: float) -> float:
        return self.offset + self.slope * x

    def generate_coefficients(self) -> list:
        return [self.offset, self.slope]


class TestQuadFunc:
    """
    TestQuadFunc is a simple representation of a linear function with slope and offset.
    It is used in the Test KnowledgeTensorPoint generation to simulate knowledge as a formula. This is a test
        generation class and should be removed when there are experts who have encoded their knowledge into
        KnowledgeTensors.
    """

    def __init__(self):
        self.c = 0.0
        self.b = 0.0
        self.a = 0.0

    @staticmethod
    def construct_offset_slope(offset: float, slope: float, second_derivative: float):
        quad_func = TestQuadFunc()
        quad_func.c = offset
        quad_func.b = slope
        quad_func.a = second_derivative
        return quad_func

    @staticmethod
    def construct_3_points(x0: float, y0: float, x1: float, y1: float, x2: float, y2: float):

        d1 = x0 * x0
        d2 = x0
        d3 = 1.0
        d4 = y0
        d5 = x1 * x1
        d6 = x1
        d7 = 1.0
        d8 = y1
        d9 = x2 * x2
        d10 = x2
        d11 = 1.0
        d12 = y2

        if x0 != 0:

            # e5 = d5 - d1 * d5 / d1
            e6 = d6 - d2 * d5 / d1
            e7 = d7 - d3 * d5 / d1
            e8 = d8 - d4 * d5 / d1

            # e9 = d9 - d1 * d9 / d1
            e10 = d10 - d2 * d9 / d1
            e11 = d11 - d3 * d9 / d1
            e12 = d12 - d4 * d9 / d1

            # f10 = e10 - e6 * e10 / e6
            f11 = e11 - e7 * e10 / e6
            f12 = e12 - e8 * e10 / e6

            c = f12 / f11
            b = (e8 - e7 * c) / e6
            a = (d4 - d2 * b - d3 * c) / d1

            return TestQuadFunc.construct_offset_slope(c, b, a)
        else:
            # e9 = d9 - d5 * d9 / d5
            e10 = d10 - d6 * d9 / d5
            e11 = d11 - d7 * d9 / d5
            e12 = d12 - d8 * d9 / d5

            c = d4 / d3
            b = (e12 - e11 * c) / e10
            a = (d8 - d6 * b - d7 * c) / d5

            return TestQuadFunc.construct_offset_slope(c, b, a)

    def eval(self, x: float) -> float:
        return self.c + self.b * x + self.a * x * x

    def generate_coefficients(self) -> list:
        return [self.c, self.b, self.a]

    def create_formatted_str(self, var_name: str) -> str:
        formatted_polynomial = ''
        coefficients = self.generate_coefficients()

        for i in range(len(coefficients)):
            single_coefficient = coefficients[i]

            if single_coefficient == 0:
                continue

            positive_coefficient = abs(single_coefficient)
            if single_coefficient >= 0:
                if i != 0:
                    formatted_polynomial += '+ '
            else:
                formatted_polynomial += '- '

            if i > 0 and positive_coefficient == 1:
                # don't show number
                pass
            elif positive_coefficient > 0.01:

                format_template = '%.5f'

                formatted_polynomial += format_template % positive_coefficient
            else:

                format_template = '%.5e'
                formatted_polynomial += format_template % positive_coefficient

            if i == 1:
                if positive_coefficient != 1.0:
                    formatted_polynomial += '*'
                formatted_polynomial += var_name
            elif i >= 2:
                if positive_coefficient != 1.0:
                    formatted_polynomial += '*'
                formatted_polynomial += var_name + '^' + str(i)

            formatted_polynomial += ' '

        return formatted_polynomial


class TestSegment:
    """
    TestSegment is a simple representation of a segment in 2D space. It is used in Test KnowledgeTensorPoint generation
    as an intermediary between a TestCurve and a TestLinearFunc. This is a test generation class and
    should be removed when there are experts who have encoded their knowledge into KnowledgeTensors.
    """

    def __init__(self, x_left: float, y_left: float, x_right: float, y_right: float):
        self.x_left = x_left
        self.y_left = y_left
        self.x_right = x_right
        self.y_right = y_right

    def linear_evaluation(self, x_val: float) -> float:
        if self.y_right == self.y_left:
            return self.y_left

        run = self.x_right - self.x_left
        rise = self.y_right - self.y_left
        slope = rise / run

        delta_x = x_val - self.x_left

        return delta_x * slope + self.y_left

    def compute_adjustment_linear_function(self, desired_segment) -> TestLinearFunc:
        """
        TestKtpCreateDirectBasedOnInputs will multiply all the inputs together to get
        the input segment. It will also have a desired_segment. It then creates a
        TestLinearFunc to convert the input segment to the desired segment.
        """
        input_segment = self

        # This is the same line
        if desired_segment.y_left == input_segment.y_left and desired_segment.y_right == input_segment.y_right:
            adjustment_linear_function = TestLinearFunc.construct_offset_slope(0.0, 0.0)

        # It is  a horizontal line before and after
        elif input_segment.y_left == input_segment.y_right:
            # This works well if desired_segment.y_left == desired_segment.y_right
            # If it is not this block will assume it is.
            #   It is difficult to create a Sloped Line from a Horizontal Line
            #   without access to another Sloped Line

            # if it is a horizontal line on zero
            if input_segment.y_left == 0.0:
                adjustment_linear_function = TestLinearFunc.construct_offset_slope(desired_segment.y_left, 0.0)
            # if it is a horizontal line not on zero
            else:
                # can either return an offset or a multiplier
                # decided to do a multiplier
                r = desired_segment.y_left / input_segment.y_left
                adjustment_linear_function = TestLinearFunc.construct_offset_slope(0.0, r)
        else:

            y2 = desired_segment.y_left
            y3 = desired_segment.y_right

            a = (y3 - y2) / (input_segment.y_right - input_segment.y_left)
            b = y2 - input_segment.y_left * a

            adjustment_linear_function = TestLinearFunc.construct_offset_slope(b, a)

        return adjustment_linear_function

    def to_linear_func(self) -> TestLinearFunc:

        dx = self.x_right - self.x_left
        dy = self.y_right - self.y_left
        slope = dy / dx
        offset = self.y_right - slope * self.x_right

        return TestLinearFunc.construct_offset_slope(offset, slope)


class TestDataContainer:

    def __init__(self):
        self.time_range = None
        self.core_name_TO_curve = {}
        self.core_name_TO_ktp_type = {}
        self.core_name_TO_inputs = {}
        self.filter_params = {}

        self.core_name_TO_is_distribution = {}
        self.user_filter = None
        self.root = None

        self.converters = {}

        self.action_TO_start_time = {}

        self.name_TO_action_TO_curve = {}

        self.counter = 123456789

    def set_filter_1(self, core_name: str, filter_dim1: str, level1: int) -> None:
        if core_name not in self.filter_params:
            self.filter_params[core_name] = TestFilterParameters(core_name)
        self.filter_params[core_name].set(filter_dim1, level1)

    def set_filter_2(self, core_name: str, filter_dim1: str, level1: int, filter_dim2: str, level2: int) -> None:
        if core_name not in self.filter_params:
            self.filter_params[core_name] = TestFilterParameters(core_name)

        self.filter_params[core_name].set(filter_dim1, level1)
        self.filter_params[core_name].set(filter_dim2, level2)

    def generate_main_filter(self, core_name: str) -> str or None:
        if core_name not in self.filter_params:
            return None

        return self.filter_params[core_name].generate_used_mloc(self.user_filter)

    def generate_not_used_filters(self, core_name: str) -> list or None:
        if core_name not in self.filter_params:
            return None

        return self.filter_params[core_name].generate_unused_filters(self.user_filter)

    def aggregate_all_from_dimensions(self) -> None:
        """
        The Life Scores will need to be aggregated into a single life score.
        This function creates a set of aggregation core_names without having to
        manually specify every aggregation.
        """

        for core_name in self.core_name_TO_curve.keys():
            if core_name not in self.core_name_TO_ktp_type:
                self.core_name_TO_ktp_type[core_name] = KtpTypeDef.dir_est

        all_core_names = list(self.core_name_TO_ktp_type.keys())
        all_core_names.sort(key=lambda core_name: '%02d' % core_name.count(MultiDimLoc.hierarchy) + '_' + core_name)
        all_core_names.reverse()

        for i in range(len(all_core_names)):
            core_name = all_core_names[i]
            start_mloc = MultiDimLoc(core_name)

            if start_mloc.has_dimension(Dimensions.core_from) is False:
                continue

            remaining = str(start_mloc.remove(Dimensions.core_from))

            start_from = start_mloc.get(Dimensions.core_from)

            iterator_from = start_from
            while True:
                mloc_at_from = remaining + MultiDimLoc.separator + iterator_from
                parent_from = MultiDimLoc.get_parent(iterator_from)

                parent_mloc = remaining + MultiDimLoc.separator + parent_from
                if parent_from == Dimensions.core_from:
                    parent_mloc = remaining

                if parent_mloc not in self.core_name_TO_ktp_type:
                    self.core_name_TO_ktp_type[parent_mloc] = KtpTypeDef.aggr
                    self.core_name_TO_inputs[parent_mloc] = []
                    self.core_name_TO_curve[parent_mloc] = self.core_name_TO_curve[core_name]

                else:

                    self.core_name_TO_curve[parent_mloc] += self.core_name_TO_curve[core_name]

                if mloc_at_from not in self.core_name_TO_inputs[parent_mloc]:
                    self.core_name_TO_inputs[parent_mloc].append(mloc_at_from)

                iterator_from = parent_from
                if parent_from == Dimensions.core_from:
                    break

    def format_curve(self, curve: TestCurve) -> str:
        return format_float(curve.y_start) + '->' + format_float(curve.y_end)

    def recursive_json_log(self, core_name: str, json_logger: JsonLogger) -> None:
        json_node = json_logger.create_node()

        json_node.set_small_child('core_name', core_name)
        json_node.set_small_child('type', self.core_name_TO_ktp_type[core_name])

        if core_name in self.core_name_TO_curve:
            json_node.set_small_child('curve', self.format_curve(self.core_name_TO_curve[core_name]))

        json_node.large_children_name = 'inputs'

        self.core_name_TO_inputs[core_name].sort()

        for input_core_name in self.core_name_TO_inputs[core_name]:
            self.recursive_json_log(input_core_name, json_logger)

        json_logger.go_to_parent()

    def log_to_json(self, file_name: str) -> None:

        json_logger = JsonLogger()

        json_logger.active_node.set_small_child('num', len(self.core_name_TO_ktp_type))
        json_logger.active_node.large_children_name = 'children'

        self.recursive_json_log(self.root, json_logger)

        json_logger.log_to_json(file_name)


class TestFilterParameters:
    """
    TestFilterParameters generates proper and improper filterable dimensions to test the CoreTensors ability
    to properly query KnowledgeTensorPoint. This is a test generation class and should be removed when
    there are experts who have encoded their knowledge into KnowledgeTensors.
    """

    def __init__(self, core_name: str):
        self.core_name = core_name
        self.core_mloc = MultiDimLoc(self.core_name)
        self.filter_dims = []
        self.filter_levels = []

    def set(self, filter_dimension: str, filter_level: int) -> None:
        self.filter_dims.append(filter_dimension)
        self.filter_levels.append(filter_level)

    def get_used_single_loc(self, user_filter: UserFilter, filter_index: int) -> str:
        raw_filter_val = user_filter.retrieve_filter_value(self.core_mloc.get(Dimensions.core_object),
                                                           self.filter_dims[filter_index])

        num_levels = MultiDimLoc.compute_num_levels(raw_filter_val)

        level = self.filter_levels[filter_index]
        if level > num_levels:
            level = num_levels

        return MultiDimLoc.filter_up_to_level(raw_filter_val, level)

    def generate_used_mloc(self, user_filter: UserFilter) -> str:
        used_mloc = ''
        for i in range(len(self.filter_dims)):
            if i != 0:
                used_mloc += MultiDimLoc.separator
            used_mloc += self.get_used_single_loc(user_filter, i)

        return used_mloc

    def gen_unused_parent(self, user_filter: UserFilter, filter_index: int) -> str or None:
        used = self.get_used_single_loc(user_filter, filter_index)
        if MultiDimLoc.is_root(used):
            return None

        return MultiDimLoc.get_parent(used)

    def gen_unused_sibling(self, user_filter: UserFilter, filter_index: int) -> str or None:
        used = self.get_used_single_loc(user_filter, filter_index)
        if MultiDimLoc.is_root(used):
            return None

        return MultiDimLoc.get_parent(used) + MultiDimLoc.hierarchy + 'unused_sibling_1'

    @staticmethod
    def gen_unused_child(user_dim_val: str) -> str:
        return user_dim_val + MultiDimLoc.hierarchy + 'unused_child_1'

    def generate_unused_filters(self, user_filter: UserFilter) -> list:

        obj = self.core_mloc.get(Dimensions.core_object)
        unused_filters = []

        mloc_so_far = ''
        for i in range(len(self.filter_dims)):
            if i < len(self.filter_dims) - 1:
                mloc_so_far += self.get_used_single_loc(user_filter, i) + MultiDimLoc.separator
            else:
                unused_parent = self.gen_unused_parent(user_filter, i)
                if unused_parent is not None:
                    unused_filters.append(mloc_so_far + unused_parent)

                unused_child = TestFilterParameters.gen_unused_child(
                    user_filter.retrieve_filter_value(obj, self.filter_dims[i]))
                if unused_child is not None:
                    unused_filters.append(mloc_so_far + unused_child)

                unused_sibling = self.gen_unused_sibling(user_filter, i)
                if unused_sibling is not None:
                    unused_filters.append(mloc_so_far + unused_sibling)

        other_values = []

        for i in range(len(self.filter_dims)):
            if i == 0:
                unused_parent = self.gen_unused_parent(user_filter, i)
                if unused_parent is not None:
                    other_values.append(unused_parent)

                unused_child = TestFilterParameters.gen_unused_child(
                    user_filter.retrieve_filter_value(obj, self.filter_dims[i]))
                if unused_child is not None:
                    other_values.append(unused_child)

                unused_sibling = self.gen_unused_sibling(user_filter, i)
                if unused_sibling is not None:
                    other_values.append(unused_sibling)

            else:
                used_loc = MultiDimLoc.separator + self.get_used_single_loc(user_filter, i)
                for j in range(len(other_values)):
                    other_values[j] += used_loc

        unused_filters.extend(other_values)

        return sorted(unused_filters)


class TestDataCreator1(TestDataContainer):
    """
    TestDataCreator1 is an instance of TestDataContainer and is used to set test values so that
    KnowledgeTensors can be generated. This is a test generation class and should be removed when
    there are experts who have encoded their knowledge into KnowledgeTensors.
    """

    def __init__(self, time_range: TimeRange, base_line: float):
        super().__init__()

        self.time_range = time_range

        self.base_line = base_line
        self.base_line_major_increase = TestCurve(base_line * 1.0, base_line * 1.2, base_line * 2.1)
        self.base_line_minor_increase = TestCurve(base_line * 1.0, base_line * 1.02, base_line * 1.04)
        self.base_line_major_decrease = TestCurve(base_line * 1.0, base_line * 0.8, base_line * 0.5)
        self.base_line_minor_decrease = TestCurve(base_line * 1.0, base_line * 0.98, base_line * 0.96)

        self.base_line_level = TestCurve(base_line, base_line, base_line)

        self.root = Name.persons_metric(MetricDef.life_score_per_year)

        self.converters = {MetricDef.money_converter: 1.0, MetricDef.death_converter: -1.0 * 10000,
                           MetricDef.time_converter: -1.0 * 10}

        self.action_TO_start_time[Dimensions.reality] = time_range.start

        self.define_environment()
        self.define_income()
        self.define_natural_non_mass_disasters()
        self.define_natural_mass_disasters()
        self.define_med()
        self.define_food()
        self.define_shelter()
        self.define_tran()

        self.define_actions()

        self.aggregate_all_from_dimensions()

    def set(self, core_name: str, core_type: str, inputs: list, y3: TestCurve or None) -> None:

        self.core_name_TO_ktp_type[core_name] = core_type
        self.core_name_TO_inputs[core_name] = inputs

        if y3 is None:
            y3 = self.core_name_TO_curve[inputs[0]]
            for i in range(1, len(inputs)):
                y3 += self.core_name_TO_curve[inputs[i]]

        self.core_name_TO_curve[core_name] = y3

    @staticmethod
    def determine_metric_per_event(core_name: str, metric: str) -> str:
        new_metric = ''
        if metric == MetricDef.money_converter:
            new_metric = MetricDef.money_per_event
        elif metric == MetricDef.death_converter:
            new_metric = MetricDef.death_per_event
        elif metric == MetricDef.time_converter:
            new_metric = MetricDef.time_per_event
        elif metric == MetricDef.number_per_year:
            new_metric = MetricDef.number_per_year

        of = core_name.replace(MultiDimLoc.separator + Dimensions.core_from, MultiDimLoc.separator + Dimensions.core_of)
        return of.replace(MultiDimLoc.separator + MetricDef.life_score_per_year, MultiDimLoc.separator + new_metric)

    def set_event(self, life_score_event: str, life_score_y3: TestCurve, n_py_start: float,
                  n_py_percent_growth_share: float, percent_starts: dict, percent_ends: dict,
                  dependencies: dict) -> None:

        n_py_breakdown = life_score_y3.split_n_py_percent(n_py_start, n_py_percent_growth_share)
        n_py = n_py_breakdown[0]
        ls_per_event = n_py_breakdown[1]

        n_py_dependencies = []
        if MetricDef.number_per_year in dependencies:
            n_py_dependencies = [dependencies[MetricDef.number_per_year]]

        number_per_year = TestDataCreator1.determine_metric_per_event(life_score_event, MetricDef.number_per_year)
        self.set(number_per_year, KtpTypeDef.dir_est, n_py_dependencies, n_py)

        self.set_filter_1(number_per_year, Dimensions.address, 2)

        metrics_per_events = ls_per_event.split_add_with_converter(self.converters, percent_starts, percent_ends)

        life_score_inputs = [number_per_year]
        for metric in self.converters.keys():
            metric_per_event = TestDataCreator1.determine_metric_per_event(life_score_event, metric)

            life_score_inputs.append(metric_per_event)

            metric_dependencies = []
            if metric in dependencies:
                metric_dependencies = [dependencies[metric]]

            self.set(metric_per_event, KtpTypeDef.dir_est, metric_dependencies, metrics_per_events[metric])

        self.set(life_score_event, KtpTypeDef.life_score_event, life_score_inputs, life_score_y3)

    def define_environment(self) -> None:
        self.set(Name.environment_metric(MetricDef.climate_change), KtpTypeDef.dir_est, [], TestCurve(1.3, 1.4, 1.7))
        self.set(Name.environment_metric(MetricDef.surface_water), KtpTypeDef.dir_est,
                 [Name.environment_metric(MetricDef.climate_change)], TestCurve(113.0, 117.0, 123.0))
        self.set(Name.environment_metric(MetricDef.ocean_temp), KtpTypeDef.dir_est,
                 [Name.environment_metric(MetricDef.climate_change)], TestCurve(43.0, 44.0, 46.0))
        self.set(Name.environment_metric(MetricDef.air_temp), KtpTypeDef.dir_est,
                 [Name.environment_metric(MetricDef.climate_change)], TestCurve(33.0, 34.0, 36.0))

        self.set(Name.environment_metric(MetricDef.surface_water), KtpTypeDef.dir_est,
                 [Name.environment_metric(MetricDef.climate_change)], TestCurve(113.0, 117.0, 123.0))
        self.set(Name.environment_metric(MetricDef.ocean_temp), KtpTypeDef.dir_est,
                 [Name.environment_metric(MetricDef.climate_change)], TestCurve(43.0, 44.0, 46.0))

    def define_income(self) -> None:
        hrs_per_year = 40.0 * 52.0

        self.set(Name.persons_annual_money_of(Of.emp_inc), KtpTypeDef.dir_mul,
                 [Name.persons_metric(MetricDef.hours_per_year), Name.persons_metric(MetricDef.money_per_hour)],
                 TestCurve(self.base_line * 3.0, self.base_line * 3.1, self.base_line * 3.2))
        self.set(Name.persons_annual_money_of(Of.emp_tax), KtpTypeDef.dir_est, [],
                 TestCurve(self.base_line * -0.5, self.base_line * -0.6, self.base_line * -0.7))
        self.set_filter_1(Name.persons_annual_money_of(Of.emp_tax), Dimensions.address, 2)

        self.set(Name.person_life_score_from(From.employment_income), KtpTypeDef.aggr,
                 [Name.persons_annual_money_of(Of.emp_inc), Name.persons_annual_money_of(Of.emp_tax)], None)

        self.set(Name.persons_metric(MetricDef.hours_per_year), KtpTypeDef.dir_est, [],
                 TestCurve(hrs_per_year, hrs_per_year, hrs_per_year))

        self.set(Name.persons_metric(MetricDef.money_per_hour), KtpTypeDef.dir_est, [],
                 self.core_name_TO_curve[Name.persons_annual_money_of(Of.emp_inc)] / hrs_per_year)
        self.set_filter_2(Name.persons_metric(MetricDef.money_per_hour), Dimensions.profession, 1, Dimensions.address,
                          2)

        self.set_event(Name.person_life_score_from(From.drowning_accidents), self.base_line_minor_increase * -0.06,
                       0.1, 0.90, {MetricDef.death_converter: 0.5}, {MetricDef.death_converter: 0.55}, {})

        self.set_event(Name.person_life_score_from(From.shark_attack_employment), self.base_line_minor_decrease * -0.07,
                       0.2, 1.05, {MetricDef.death_converter: 0.4}, {MetricDef.death_converter: 0.3}, {})

    def define_natural_non_mass_disasters(self) -> None:

        self.set_event(Name.person_life_score_from(From.shark_attack_nature), self.base_line_minor_increase * -0.009,
                       0.1, 0.90, {MetricDef.death_converter: 0.4}, {MetricDef.death_converter: 0.35}, {})

        self.set_event(Name.person_life_score_from(From.lightning_strike), self.base_line_minor_increase * -0.006,
                       0.1, 0.90, {MetricDef.death_converter: 0.6}, {MetricDef.death_converter: 0.55}, {})

    def mass_dis_helper(self, life_score: str, all_base_line: TestCurve, level_to_percents: dict, level_to_n_py: dict,
                        dependency: str) -> None:

        dependencies = {}
        if dependency != '':
            dependencies = {MetricDef.number_per_year: dependency}

        for level in level_to_percents.keys():
            ls_name = life_score + MultiDimLoc.hierarchy + str(level)
            level_curve = all_base_line * level_to_percents[level]

            n_py = level_to_n_py[level]

            self.set_event(ls_name, level_curve, n_py,
                           0.99, {MetricDef.death_converter: 0.2}, {MetricDef.death_converter: 0.15}, dependencies)

    def define_natural_mass_disasters(self) -> None:

        nature_baseline = TestCurve(self.base_line * -0.1, self.base_line * -0.2, self.base_line * -0.9)

        self.mass_dis_helper(Name.person_life_score_from(From.flood), nature_baseline * 0.27,
                             {2: 0.2, 3: 0.3, 4: 0.4, 5: 0.1}, {2: 0.6, 3: 0.1, 4: 0.01, 5: 0.001},
                             Name.environment_metric(MetricDef.surface_water))

        self.mass_dis_helper(Name.person_life_score_from(From.hurricane), nature_baseline * 0.28,
                             {2: 0.2, 3: 0.3, 4: 0.4, 5: 0.1}, {2: 0.6, 3: 0.1, 4: 0.01, 5: 0.001},
                             Name.environment_metric(MetricDef.ocean_temp))
        self.mass_dis_helper(Name.person_life_score_from(From.tornado), nature_baseline * 0.29,
                             {2: 0.2, 3: 0.3, 4: 0.4, 5: 0.1}, {2: 0.6, 3: 0.1, 4: 0.01, 5: 0.001},
                             Name.environment_metric(MetricDef.air_temp))

        self.mass_dis_helper(Name.person_life_score_from(From.earth_quake), self.base_line_minor_decrease * 0.05,
                             {4: 0.2, 5: 0.3, 6: 0.4, 7: 0.1}, {4: 0.6, 5: 0.1, 6: 0.01, 7: 0.001}, '')

    def define_med(self) -> None:

        for i in range(1, 6):
            r = -0.001 * i + -0.001
            core_name = Name.person_life_score_from(From.cancer + MultiDimLoc.hierarchy + 'subtype_' + str(i))
            self.set_event(core_name, self.base_line_minor_increase * r, 0.001, 0.90, {MetricDef.death_converter: 0.3},
                           {MetricDef.death_converter: 0.25}, {})

            self.set_filter_1(core_name, Dimensions.genetic, 1)

        self.set(Name.persons_metric(MetricDef.blood_pressure), KtpTypeDef.dir_est, [], TestCurve(120.0, 125.0, 130.0))

        for i in range(1, 5):
            r = -0.0015 * i + -0.002
            core_name = Name.person_life_score_from(From.heart_attack + MultiDimLoc.hierarchy + 'subtype_' + str(i))

            self.set_event(core_name, self.base_line_major_increase * r, 0.001, 0.90, {MetricDef.death_converter: 0.3},
                           {MetricDef.death_converter: 0.25},
                           {MetricDef.number_per_year: Name.persons_metric(MetricDef.blood_pressure)})

            self.set_filter_1(core_name, Dimensions.smoker, 1)

        self.set(Name.persons_metric(MetricDef.bmi), KtpTypeDef.dir_est, [], TestCurve(24.0, 27.0, 29.0))
        self.set_filter_1(Name.persons_metric(MetricDef.bmi), Dimensions.address, 3)

        for i in range(1, 5):
            r = -0.002 * i + -0.001
            core_name = Name.person_life_score_from(From.diabetes + MultiDimLoc.hierarchy + 'subtype_' + str(i))

            self.set_event(core_name, self.base_line_minor_increase * r, 0.001, 0.90, {MetricDef.death_converter: 0.3},
                           {MetricDef.death_converter: 0.25},
                           {MetricDef.number_per_year: Name.persons_metric(MetricDef.bmi)})

            self.set_filter_1(core_name, Dimensions.genetic, 1)

        self.set_event(Name.person_life_score_from(From.covid), self.base_line_minor_decrease * -0.09, 0.001, 0.90,
                       {MetricDef.death_converter: 0.05}, {MetricDef.death_converter: 0.05}, {})
        self.set_event(Name.person_life_score_from(From.flu), self.base_line_minor_increase * -0.07, 0.001, 0.90,
                       {MetricDef.death_converter: 0.03}, {MetricDef.death_converter: 0.03}, {})

        self.set(Name.person_life_score_from(From.med_insurance), KtpTypeDef.dir_est, [],
                 self.base_line_major_increase * -0.1)
        self.set_filter_1(Name.person_life_score_from(From.med_insurance), Dimensions.address, 2)

        self.set(Name.person_life_score_from(From.med_tax), KtpTypeDef.dir_est,
                 [Name.persons_annual_money_of(Of.emp_inc)], self.base_line_minor_increase * -0.02)
        self.set_filter_1(Name.person_life_score_from(From.med_tax), Dimensions.address, 1)

    def define_food(self) -> None:
        self.set(Name.economic_cpi_of(Of.food), KtpTypeDef.dir_est, [Name.environment_metric(MetricDef.climate_change)],
                 TestCurve(0.9, 1.1, 1.4))
        self.set(Name.persons_annual_money_of(Of.food_dir), KtpTypeDef.dir_est,
                 [Name.persons_annual_money_of(Of.emp_inc)], self.base_line_minor_increase * -0.3)

        self.set(Name.persons_annual_money_of(Of.food_tax), KtpTypeDef.dir_est,
                 [Name.persons_annual_money_of(Of.emp_inc)], self.base_line_minor_increase * -0.09)

        self.set(Name.person_life_score_from(From.food), KtpTypeDef.aggr,
                 [Name.persons_annual_money_of(Of.food_dir), Name.persons_annual_money_of(Of.food_tax)], None)

    def define_shelter(self) -> None:

        self.set(Name.person_life_score_from(From.rent), KtpTypeDef.dir_est, [], self.base_line_minor_increase * -0.9)
        self.set_filter_1(Name.person_life_score_from(From.rent), Dimensions.address, 3)

        self.set(Name.person_life_score_from(From.renter_insurance), KtpTypeDef.dir_est, [],
                 self.base_line_minor_increase * -0.01)
        self.set_filter_1(Name.person_life_score_from(From.renter_insurance), Dimensions.address, 2)

        self.set(Name.person_life_score_from(From.electricity), KtpTypeDef.dir_est, [],
                 self.base_line_minor_increase * -0.05)
        self.set_filter_1(Name.person_life_score_from(From.electricity), Dimensions.address, 3)

        self.set(Name.person_life_score_from(From.water), KtpTypeDef.dir_est, [], self.base_line_minor_increase * -0.03)
        self.set_filter_1(Name.person_life_score_from(From.water), Dimensions.address, 3)

        self.set_event(Name.person_life_score_from(From.house_fire), self.base_line_major_decrease * -0.012, 0.001,
                       0.90, {MetricDef.death_converter: 0.01}, {MetricDef.death_converter: 0.01}, {})

        self.set_event(Name.person_life_score_from(From.pipe_burst), self.base_line_major_decrease * -0.004, 0.001,
                       0.90, {MetricDef.death_converter: 0.001}, {MetricDef.death_converter: 0.001}, {})

        self.set(Name.person_life_score_from(From.shelter_tax), KtpTypeDef.dir_est,
                 [Name.persons_annual_money_of(Of.emp_inc)], self.base_line_minor_increase * -0.02)
        self.set_filter_1(Name.person_life_score_from(From.shelter_tax), Dimensions.address, 1)

    def define_tran(self) -> None:
        self.set(Name.person_life_score_from(From.car_tran), KtpTypeDef.dir_est, [],
                 self.base_line_minor_increase * -0.1)
        self.set(Name.person_life_score_from(From.car_maintenance), KtpTypeDef.dir_est, [],
                 self.base_line_minor_increase * -0.1)
        self.set(Name.person_life_score_from(From.car_fuel), KtpTypeDef.dir_est, [],
                 self.base_line_minor_increase * -0.1)

        self.set_filter_1(Name.person_life_score_from(From.car_tran), Dimensions.address, 1)
        self.set_filter_1(Name.person_life_score_from(From.car_maintenance), Dimensions.address, 3)
        self.set_filter_1(Name.person_life_score_from(From.car_fuel), Dimensions.address, 2)

        self.set_event(Name.person_life_score_from(From.car_accident), self.base_line_major_decrease * -0.025, 0.001,
                       0.90, {MetricDef.death_converter: 0.05}, {MetricDef.death_converter: 0.03}, {})

        self.set(Name.person_life_score_from(From.transportation_tax), KtpTypeDef.dir_est,
                 [Name.persons_annual_money_of(Of.emp_inc)], self.base_line_minor_increase * -0.02)

    def action_set(self, action: str, core_name: str, start_multiplier: float, mid_multplier: float,
                   end_multiplier: float) -> None:

        normal_y3 = self.core_name_TO_curve[core_name]

        quad = normal_y3.create_quadratic(self.time_range.start - self.time_range.start,
                                          self.time_range.end - self.time_range.start)

        action_start_time = self.action_TO_start_time[action]
        action_end_time = self.time_range.end
        action_mid_time = (action_start_time + action_end_time) / 2

        start_val = quad.eval(action_start_time - self.time_range.start)
        mid_val = quad.eval(action_mid_time - self.time_range.start)
        end_val = quad.eval(action_end_time - self.time_range.start)

        new_start_val = start_val * start_multiplier
        new_mid_val = mid_val * mid_multplier
        new_end_val = end_val * end_multiplier

        new_y3 = TestCurve(new_start_val, new_mid_val, new_end_val)

        if core_name not in self.name_TO_action_TO_curve:
            self.name_TO_action_TO_curve[core_name] = {}

        self.name_TO_action_TO_curve[core_name][action] = new_y3

        if Dimensions.reality not in self.name_TO_action_TO_curve[core_name]:
            self.name_TO_action_TO_curve[core_name][Dimensions.reality] = self.core_name_TO_curve[core_name]

    def define_actions(self) -> None:

        action1 = Dimensions.action + MultiDimLoc.hierarchy + 'Hypothetical_Action_01'
        self.action_TO_start_time[action1] = self.time_range.present_time + 1
        self.action_set(action1, Name.person_life_score_from(From.transportation_tax), 2.1, 2.0, 1.9)
        self.action_set(action1, Name.environment_metric(MetricDef.climate_change), 1.0, 0.99, 0.96)

        action2 = Dimensions.action + MultiDimLoc.hierarchy + 'Hypothetical_Action_02'
        self.action_TO_start_time[action2] = self.time_range.present_time + 1
        self.action_set(action2, Name.person_life_score_from(From.transportation_tax), 1.3, 1.2, 1.1)
        self.action_set(action2, Name.persons_annual_money_of(Of.emp_tax), 1.3, 1.2, 1.1)
        self.action_set(action2, Name.environment_metric(MetricDef.climate_change), 1.0, 0.96, 0.90)


class TestKtpCreateAggregator:
    """
    TestKtpCreateAggregator creates simple Aggregation KnowledgeTensorPoints. This is a test generation class and
    should be removed when there are experts who have encoded their knowledge into KnowledgeTensors.
    """

    def __init__(self, parent, core_name: str, inputs: list):
        self.parent = parent
        self.core_name = core_name
        self.inputs = inputs
        self.inputs.sort()

    def populate_kt(self, file, percent_of_action: float) -> None:
        mlocs = [self.core_name, KtpTypeDef.aggr]
        mlocs.extend(self.parent.create_lb_up_res(self.core_name))

        file.write(Name.mloc(mlocs) + '\n')

        file.write('\treturn ')
        for i in range(len(self.inputs)):
            if i != 0:
                file.write(' + ')
            file.write("get(time_index,'" + self.inputs[i] + "')")

        file.write('\n\n')


class TestKtpCreateDirectMultiplication:
    """
    TestKtpCreateAggregator creates KnowledgeTensorPoint for a Direct computation that is a multiplication of 2
    core_names. This is a test generation class and should be removed when there are experts who
    have encoded their knowledge into KnowledgeTensors.
    """

    def __init__(self, parent, core_name: str, inputs: list):
        self.core_name = core_name
        self.parent = parent

        self.inputs = inputs
        self.inputs.sort()

    def populate_kt(self, file, percent_of_action: float) -> None:

        mlocs = [self.core_name, KtpTypeDef.dir_est, Dimensions.start_time + MultiDimLoc.hierarchy + str(0),
                 Dimensions.end_time + MultiDimLoc.hierarchy + str(10000)]
        mlocs.extend(self.parent.create_lb_up_res(self.core_name))

        file.write(Name.mloc(mlocs) + '\n')

        file.write('\treturn ')
        for i in range(len(self.inputs)):
            if i != 0:
                file.write(' * ')
            file.write("get(time_index,'" + self.inputs[i] + "')")

        file.write('\n\n')


class TestKtpCreateLifeScoreEvent:
    """
    TestKtpCreateAggregator creates KnowledgeTensorPoint for a LifeScoreEvent. This involves a metric expansion. This
    is a test generation class and should be removed when there are experts who have encoded their knowledge into
    KnowledgeTensors.
    """

    def __init__(self, parent, core_name: str, inputs: list):
        self.core_name = core_name
        self.parent = parent
        self.m_TO_input = {}
        for input_core_name in inputs:
            m = MultiDimLoc(input_core_name)
            self.m_TO_input[m.get(Dimensions.core_metric)] = input_core_name

    def populate_kt(self, file, percent_of_action: float) -> None:

        mlocs = [self.core_name, KtpTypeDef.life_score_event]
        mlocs.extend(self.parent.create_lb_up_res(self.core_name))

        file.write(Name.mloc(mlocs) + '\n')

        file.write("\tn_py = get(time_index,'" + self.m_TO_input[MetricDef.number_per_year] + "') " + '\n')

        entries = []

        if MetricDef.death_per_event in self.m_TO_input:
            file.write("\td_pe = get(time_index,'" + self.m_TO_input[MetricDef.death_per_event] + "') " + '\n')
            file.write("\td_conv = get(time_index,'" + Name.persons_metric(MetricDef.death_converter) + "') " + '\n')
            entries.append('d_pe * d_conv')

        if MetricDef.money_per_event in self.m_TO_input:
            file.write("\tm_pe = get(time_index,'" + self.m_TO_input[MetricDef.money_per_event] + "') " + '\n')
            file.write("\tm_conv = get(time_index,'" + Name.persons_metric(MetricDef.money_converter) + "') " + '\n')
            entries.append('m_pe * m_conv')

        if MetricDef.time_per_event in self.m_TO_input:
            file.write("\tt_pe = get(time_index,'" + self.m_TO_input[MetricDef.time_per_event] + "') " + '\n')
            file.write("\tt_conv = get(time_index,'" + Name.persons_metric(MetricDef.time_converter) + "') " + '\n')
            entries.append('t_pe * t_conv')

        file.write('\treturn  n_py*( ' + ' + '.join(entries) + ' ) ' + '\n')
        file.write('\n')


class TestKtpCreateDirectBasedOnTime:
    """
    TestKtpCreateDirectBasedOnTime creates a Direct KnowledgeTensorPoint that uses time as its input. This is a test
    generation class and should be removed when there are experts who have encoded their knowledge into KnowledgeTensors
    """

    def __init__(self, parent, num_shadows: int, core_name: str, action_to_truth_curve: dict):
        self.parent = parent
        self.num_shadows = num_shadows
        self.core_name = core_name
        self.action_TO_truth_curve = action_to_truth_curve

        self.action_TO_segment_variations = {}

        for action_to_process in action_to_truth_curve.keys():
            truth_segment = action_to_truth_curve[action_to_process]

            self.action_TO_segment_variations[action_to_process] = []

            segment_variations = self.action_TO_segment_variations[action_to_process]

            for i in range(self.num_shadows):
                percent_start = float(self.parent.parent.counter % 11) / 11
                percent_mid = float(self.parent.parent.counter % 13) / 13
                percent_end = float(self.parent.parent.counter % 17) / 17

                variation_segment = truth_segment.create_variation(percent_start, percent_mid, percent_end)

                segment_variations.append(variation_segment)

                self.parent.parent.counter += 127

    def create_single_ktp(self, file, action_to_process: str, segment: TestCurve) -> None:

        start_time = self.parent.parent.time_range.start
        end_time = self.parent.parent.time_range.end

        if action_to_process != Dimensions.reality:
            start_time = self.parent.parent.action_TO_start_time[action_to_process]

        polynomial_to_process = segment.create_quadratic(start_time - self.parent.parent.time_range.start,
                                                         end_time - self.parent.parent.time_range.start)

        mlocs = [self.core_name, action_to_process, KtpTypeDef.dir_est,
                 Dimensions.start_time + MultiDimLoc.hierarchy + format_float(start_time),
                 Dimensions.end_time + MultiDimLoc.hierarchy + format_float(end_time)]
        mlocs.extend(self.parent.create_lb_up_res(self.core_name))

        base_mloc = Name.mloc(mlocs)

        used_mloc_full = base_mloc

        used_mloc = self.parent.parent.generate_main_filter(self.core_name)
        if used_mloc is not None:
            used_mloc_full = Name.mloc([base_mloc, used_mloc])

        file.write(used_mloc_full + '\n')
        file.write('\tt = get_time(time_index) - ' + format_float(self.parent.parent.time_range.start) + '\n')
        file.write('\treturn ' + polynomial_to_process.create_formatted_str('t') + '\n')
        file.write('\n')

        not_used_list = self.parent.parent.generate_not_used_filters(self.core_name)
        if not_used_list is not None:
            for not_used in not_used_list:
                not_used_mloc_full = base_mloc + MultiDimLoc.separator + not_used
                file.write(not_used_mloc_full + '\n')
                file.write('\t# KTP is not supposed to be selected by the CoreTensor logic\n')
                file.write('\treturn crash_error_core_tensor()\n')
                file.write('\n')

    def create_single_ktp_distribution(self, file, action_to_process: str, segment: TestCurve) -> None:

        start_time = self.parent.parent.time_range.start
        end_time = self.parent.parent.time_range.end

        if action_to_process != Dimensions.reality:
            start_time = self.parent.parent.action_TO_start_time[action_to_process]

        mlocs = [self.core_name, action_to_process, KtpTypeDef.dir_est,
                 Dimensions.start_time + MultiDimLoc.hierarchy + format_float(start_time),
                 Dimensions.end_time + MultiDimLoc.hierarchy + format_float(end_time)]
        mlocs.extend(self.parent.create_lb_up_res(self.core_name))

        base_mloc = Name.mloc(mlocs)

        used_mloc_full = base_mloc

        used_mloc = self.parent.parent.generate_main_filter(self.core_name)
        if used_mloc is not None:
            used_mloc_full = Name.mloc([base_mloc, used_mloc])

        segment_mean = (segment.y_start + segment.y_mid + segment.y_end) / 3

        uniform_lb = segment_mean * 0.9
        uniform_ub = segment_mean * 1.1
        if segment_mean < 0:
            uniform_lb = segment_mean * 1.1
            uniform_ub = segment_mean * 0.9

        lb = mlocs[-3].split(MultiDimLoc.hierarchy)[1]
        ub = mlocs[-2].split(MultiDimLoc.hierarchy)[1]

        res = mlocs[-1].split(MultiDimLoc.hierarchy)[1]

        file.write(used_mloc_full + '\n')

        file.write('\treturn uniform_dist(' + format_float(uniform_lb) + ',' + format_float(
            uniform_ub) + ',' + lb + ',' + ub + ',' + res + ')')
        file.write('\n')

    def populate_kt(self, file, percent_of_action: float) -> None:
        all_actions = list(self.action_TO_truth_curve.keys())
        all_actions.sort()

        for action_to_process in all_actions:
            segments_of_action = self.action_TO_segment_variations[action_to_process]

            segment_index = self.parent.parent.counter % len(segments_of_action)

            if self.core_name in self.parent.parent.core_name_TO_is_distribution:
                self.create_single_ktp_distribution(file, action_to_process, segments_of_action[segment_index])
            else:
                self.create_single_ktp(file, action_to_process, segments_of_action[segment_index])

            self.parent.parent.counter += 127


class TestKtpCreateDirectBasedOnInputs:
    """
    TestKtpCreateDirectBasedOnInputs creates a Direct KnowledgeTensorPoint that uses other core_names as inputs.
    This is a test generation class and should be removed when there are experts who have encoded
    their knowledge into KnowledgeTensors.
    """

    def __init__(self, parent, num_shadows: int, core_name: str, action_to_truth_curve: dict, direct_inputs: list,
                 retrieve_curve):

        self.parent = parent

        self.num_shadows = num_shadows
        self.core_name = core_name
        self.direct_inputs = direct_inputs
        self.direct_inputs.sort()

        self.action_TO_truth_curve = action_to_truth_curve

        self.action_TO_linear_func_variations = {}

        for action_to_process in self.action_TO_truth_curve.keys():
            desired_curve = self.action_TO_truth_curve[action_to_process]
            accumulated_input_curve = retrieve_curve(self.direct_inputs[0], action_to_process)

            for i in range(1, len(self.direct_inputs)):
                input_curve = retrieve_curve(self.direct_inputs[1], action_to_process)
                accumulated_input_curve = accumulated_input_curve * input_curve

            action_start_time = self.parent.parent.action_TO_start_time[action_to_process]
            action_end_time = self.parent.parent.time_range.end

            input_segment = accumulated_input_curve.create_segment(action_start_time, 0.5, action_end_time, 0.5)

            self.action_TO_linear_func_variations[action_to_process] = []

            for i in range(self.num_shadows):
                left_multiplier = 0.5
                right_multiplier = 0.5

                desired_segment = desired_curve.create_segment(action_start_time, left_multiplier, action_end_time,
                                                               right_multiplier)
                converter_func = input_segment.compute_adjustment_linear_function(desired_segment)

                self.action_TO_linear_func_variations[action_to_process].append(converter_func)

    @staticmethod
    def create_var_name(str_mloc: str):
        all_single_dim_locs = str_mloc.split(MultiDimLoc.separator)

        new_var_name = all_single_dim_locs[0].split(MultiDimLoc.hierarchy)[-1]

        for i in range(1, len(all_single_dim_locs)):
            new_var_name += '_' + all_single_dim_locs[i].split(MultiDimLoc.hierarchy)[-1]
        return new_var_name

    def create_single_ktp(self, file, action_to_process: str, polynomial_to_process: TestLinearFunc) -> None:

        start_time = self.parent.parent.time_range.start
        end_time = self.parent.parent.time_range.end

        if action_to_process != Dimensions.reality:
            start_time = self.parent.parent.action_TO_start_time[action_to_process]

        mlocs = [self.core_name, action_to_process, KtpTypeDef.dir_est,
                 Dimensions.start_time + MultiDimLoc.hierarchy + format_float(start_time),
                 Dimensions.end_time + MultiDimLoc.hierarchy + format_float(end_time)]
        mlocs.extend(self.parent.create_lb_up_res(self.core_name))

        file.write(Name.mloc(mlocs) + '\n')

        new_vars = []
        for direct_input in self.direct_inputs:
            new_var_name = TestKtpCreateDirectBasedOnInputs.create_var_name(direct_input)
            new_vars.append(new_var_name)
            file.write('\t' + new_var_name + " = get(time_index,'" + direct_input + "')" + '\n')

        temp_input = '*'.join(new_vars)

        temp_input_name = 'prod'
        if len(self.direct_inputs) == 1:
            temp_input_name = new_vars[0]
        else:
            file.write('\tprod = ' + temp_input + '\n')

        file.write('\treturn ' + polynomial_to_process.create_formatted_str(temp_input_name) + '\n')
        file.write('\n')

    def populate_kt(self, file, percent_action: float) -> None:
        all_actions = sorted(list(self.action_TO_truth_curve.keys()))
        all_actions.sort()

        for action_to_process in all_actions:
            possible_linear_funcs = self.action_TO_linear_func_variations[action_to_process]

            variation_index = self.parent.parent.counter % len(possible_linear_funcs)

            self.create_single_ktp(file, action_to_process, possible_linear_funcs[variation_index])

            self.parent.parent.counter += 127


class TestKnowledgeTensorCreator:
    """
    TestKnowledgeTensorCreator inputs the TestDataContainer is the starting point for
    creating test KnowledgeTensors. This is a test generation class and should be
    removed when there are experts who have encoded their knowledge into KnowledgeTensors.
    """

    def __init__(self, parent: TestDataContainer):
        self.parent = parent
        self.ktp_generator = {}
        self.num_shadows = 10

        for core_name in self.parent.core_name_TO_ktp_type.keys():
            core_type = self.parent.core_name_TO_ktp_type[core_name]

            core_inputs = []
            if core_name in self.parent.core_name_TO_inputs:
                core_inputs = self.parent.core_name_TO_inputs[core_name]

            if core_type == KtpTypeDef.aggr:
                self.ktp_generator[core_name] = TestKtpCreateAggregator(self, core_name, core_inputs)
            elif core_type == KtpTypeDef.dir_est or core_type == KtpTypeDef.dir_meas:
                segments_of_alternate_actions = {}
                if core_name in self.parent.name_TO_action_TO_curve:
                    segments_of_alternate_actions = self.parent.name_TO_action_TO_curve[core_name]
                else:
                    segments_of_alternate_actions[Dimensions.reality] = self.parent.core_name_TO_curve[core_name]

                if len(core_inputs) == 0:
                    self.ktp_generator[core_name] = TestKtpCreateDirectBasedOnTime(self, self.num_shadows, core_name,
                                                                                   segments_of_alternate_actions)
                else:
                    self.ktp_generator[core_name] = TestKtpCreateDirectBasedOnInputs(self, self.num_shadows, core_name,
                                                                                     segments_of_alternate_actions,
                                                                                     core_inputs, self.get_curve)
            elif core_type == KtpTypeDef.dir_mul:
                self.ktp_generator[core_name] = TestKtpCreateDirectMultiplication(self, core_name, core_inputs)
            elif core_type == KtpTypeDef.life_score_event:
                self.ktp_generator[core_name] = TestKtpCreateLifeScoreEvent(self, core_name, core_inputs)

    def get_curve(self, core_name: str, action: str) -> TestCurve:
        if core_name in self.parent.name_TO_action_TO_curve:

            action_to_curve = self.parent.name_TO_action_TO_curve[core_name]

            if action in action_to_curve:
                return action_to_curve[action]
            else:
                return action_to_curve[Dimensions.reality]

        else:
            return self.parent.core_name_TO_curve[core_name]

    @staticmethod
    def create_lb_up_res_for_metric_conversion_ktp(d: float) -> str:

        lower_bound = d
        upper_bound = d

        if lower_bound > 0:
            lower_bound = lower_bound * 0.9
            upper_bound = upper_bound * 1.1
        else:
            lower_bound = lower_bound * 1.1
            upper_bound = upper_bound * 0.9

        raw_resolution = abs(upper_bound - lower_bound) / 20
        clean = TestKnowledgeTensorCreator.clean_resolution(raw_resolution)

        lower_bound = lower_bound / clean
        lower_bound = math.floor(lower_bound)
        lower_bound = lower_bound * clean

        upper_bound = upper_bound / clean
        upper_bound = math.floor(upper_bound)
        upper_bound = upper_bound * clean

        mlocs = [Dimensions.lower_bound + MultiDimLoc.hierarchy + format_float(lower_bound),
                 Dimensions.upper_bound + MultiDimLoc.hierarchy + format_float(upper_bound),
                 Dimensions.resolution + MultiDimLoc.hierarchy + format_float(clean)]

        return Name.mloc(mlocs)

    def create_metric_conversion_ktp(self, file) -> None:

        start_t = self.parent.time_range.start
        end_t = self.parent.time_range.end

        template = MultiDimLoc.separator + KtpTypeDef.dir_est + MultiDimLoc.separator
        template += Dimensions.start_time + MultiDimLoc.hierarchy + format_float(start_t) + MultiDimLoc.separator
        template += Dimensions.end_time + MultiDimLoc.hierarchy + format_float(end_t) + MultiDimLoc.separator

        file.write(Name.persons_metric(MetricDef.money_converter) + template
                   + TestKnowledgeTensorCreator.create_lb_up_res_for_metric_conversion_ktp(
                   self.parent.converters[MetricDef.money_converter]) + '\n')
        file.write('\treturn ' + format_float(self.parent.converters[MetricDef.money_converter]) + '\n\n')

        file.write(Name.persons_metric(MetricDef.death_converter) + template +
                   TestKnowledgeTensorCreator.create_lb_up_res_for_metric_conversion_ktp(
                   self.parent.converters[MetricDef.death_converter]) + '\n')
        file.write('\treturn ' + format_float(self.parent.converters[MetricDef.death_converter]) + '\n\n')

        file.write(Name.persons_metric(MetricDef.time_converter) + template +
                   TestKnowledgeTensorCreator.create_lb_up_res_for_metric_conversion_ktp(
                   self.parent.converters[MetricDef.time_converter]) + '\n')
        file.write('\treturn ' + format_float(self.parent.converters[MetricDef.time_converter]) + '\n\n')

    def generate_kt(self, directory: str) -> None:
        if directory not in os.listdir():
            os.mkdir(directory)

        all_core_names = sorted(self.ktp_generator.keys())

        for expert_index in range(self.num_shadows * 2):
            file_name = os.path.join(directory,
                                     'expert_Dr_' + chr(ord('A') + expert_index) + '_knowledge_tensor' + '.kt')

            file = open(file_name, 'w')

            size = int(0.8 * len(all_core_names))

            selected_indexes = []

            while len(selected_indexes) < size:
                selected_index = self.parent.counter % len(all_core_names)
                self.parent.counter += 127

                if selected_index not in selected_indexes:
                    selected_indexes.append(selected_index)

            ktp_in_kt = []

            for i in selected_indexes:
                ktp_in_kt.append(all_core_names[i])

            ktp_in_kt.sort()

            self.create_metric_conversion_ktp(file)

            for ktp in ktp_in_kt:
                gen = self.ktp_generator[ktp]
                gen.populate_kt(file, 0.7)

            file.close()

    @staticmethod
    def clean_resolution(raw_resolution: float) -> float:
        for i in range(10, -10, -1):
            temp_res = math.pow(10, i) * 5
            if temp_res < raw_resolution:
                return temp_res

            temp_res = math.pow(10, i) * 2
            if temp_res < raw_resolution:
                return temp_res

            temp_res = math.pow(10, i) * 1
            if temp_res < raw_resolution:
                return temp_res

    def create_lb_up_res(self, core_name: str) -> list:

        curve = self.parent.core_name_TO_curve[core_name]

        lower_bound = curve.calculate_lower_val()
        upper_bound = curve.calculate_upper_val()

        num_bins = 20

        if core_name == self.parent.root:
            lower_bound = -10000
            upper_bound = +10000
            num_bins = 100

        if lower_bound > 0:
            lower_bound = lower_bound * 0.8
            upper_bound = upper_bound * 1.2

            raw_resolution = abs(upper_bound - lower_bound) / num_bins
            clean = TestKnowledgeTensorCreator.clean_resolution(raw_resolution)

            lower_bound = lower_bound / clean
            lower_bound = math.floor(lower_bound)
            lower_bound = lower_bound * clean

            upper_bound = upper_bound / clean
            upper_bound = math.ceil(upper_bound)
            upper_bound = upper_bound * clean
        else:
            lower_bound = lower_bound * 1.2
            upper_bound = upper_bound * 0.8

            raw_resolution = abs(upper_bound - lower_bound) / num_bins
            clean = TestKnowledgeTensorCreator.clean_resolution(raw_resolution)

            lower_bound = lower_bound / clean
            lower_bound = math.floor(lower_bound)
            lower_bound = lower_bound * clean

            upper_bound = upper_bound / clean
            upper_bound = math.ceil(upper_bound)
            upper_bound = upper_bound * clean

        mlocs = [Dimensions.lower_bound + MultiDimLoc.hierarchy + format_float(lower_bound),
                 Dimensions.upper_bound + MultiDimLoc.hierarchy + format_float(upper_bound),
                 Dimensions.resolution + MultiDimLoc.hierarchy + format_float(clean)]

        return mlocs


class TestBlockChainCreator:
    """
    TestBlockChainCreator creates a test BlockChain of transactions.  This is a test generation class and should
    be removed when there are experts who have encoded their knowledge into KnowledgeTensors.
    """

    @staticmethod
    def generate(desired_kt_hash_tc_values: dict, file_name: str) -> list:
        hash_format = '%08x'

        user_to_kt_hash_purchase = {}

        purchase_bc = []

        user_id = 1000
        for kt_hash in sorted(desired_kt_hash_tc_values.keys()):
            for i in range(desired_kt_hash_tc_values[kt_hash]):
                user_to_kt_hash_purchase[user_id] = kt_hash

                transaction = {'transaction_type': 'knowledge_tensor_purchase', 'tc_holder': str(user_id)}

                sh = SecureHash()
                sh.update(str(user_id))

                transaction['tc_holder_public_key'] = hash_format % sh.digest()
                transaction['knowledge_tensor_hash'] = kt_hash

                purchase_bc.append(transaction)

                user_id += 1

        awarded_tc_bc = []

        for user in user_to_kt_hash_purchase.keys():
            for i in range(1, 11):
                transaction = {'transaction_type': 'award_tc', 'sub_foundation': str(i)}

                sh_foundation = SecureHash()
                sh_foundation.update(str(i))
                transaction['sub_foundation_public_key'] = hash_format % sh_foundation.digest()

                sh_user = SecureHash()
                sh_user.update(str(user))

                transaction['tc_holder'] = str(user)
                transaction['tc_holder_public_key'] = hash_format % user

                awarded_tc_bc.append(transaction)

        bc = []
        bc.extend(awarded_tc_bc)
        bc.extend(purchase_bc)

        time = 2024.001
        for i in range(len(bc)):
            bc[i]['transaction_time'] = format_float(time)
            time += 0.001

        json_logger = JsonLogger()
        json_logger.active_node.large_children_name = 'transactions'

        json_logger.set_small_child('total_transactions', len(bc))
        for transaction in bc:
            json_node = json_logger.create_node()

            for key in sorted(transaction.keys()):
                json_node.set_small_child(key, transaction[key])

            json_logger.go_to_parent()

        json_logger.log_to_json(file_name)

        return bc


def main():
    directory = 'test'

    if directory not in os.listdir():
        os.mkdir(directory)

    print('creating test UserFilter')
    user_filter = UserFilter()
    user_filter.birthday = 2000.0
    user_filter.set(Objects.person, MultiDimLoc.build_mloc_str([Dimensions.profession, 'under_water_basket_weaver']))
    user_filter.set(Objects.person,
                    MultiDimLoc.build_mloc_str([Dimensions.address, 'enmai_land', 'enmai_state', 'enmai_city']))
    user_filter.set(Objects.person, MultiDimLoc.build_mloc_str([Dimensions.ethnicity, 'ethnicity_1']))
    user_filter.set(Objects.person, MultiDimLoc.build_mloc_str([Dimensions.gender, 'gender_1']))

    user_filter.set(Objects.person, MultiDimLoc.build_mloc_str([Dimensions.smoker, 'smoker_status_1']))
    user_filter.set(Objects.person, MultiDimLoc.build_mloc_str([Dimensions.genetic, 'gene_marker_1']))

    user_filter.log_to_json(os.path.join(directory, 'user_filter.json'))

    time_range = TimeRange(2020.0, 2040.0, 1.0, 2024.0)

    print('creating test KnowledgeTensor phase 1')
    test_phase1 = TestDataCreator1(time_range, 10000.0)
    test_phase1.user_filter = user_filter
    test_phase1.log_to_json(os.path.join(directory, 'test_knowledge_tensor_high_level.json'))

    print('creating test KnowledgeTensor phase 2')
    utg_phase2 = TestKnowledgeTensorCreator(test_phase1)
    utg_phase2.generate_kt(directory)

    print('creating UniversalKnowledgeTensor')
    ut = UniversalKnowledgeTensor()
    ut.time_range = time_range
    ut.aggregate_into_distribution = True
    ut.user_filter = user_filter
    ut.discount_rate = 0.98

    desired_kt_hash_tc_values = {}

    print('loading UniversalKnowledgeTensor with KnowledgeTensors')
    for file_name in os.listdir(directory):
        if file_name.endswith('.kt') is False:
            continue

        r = open(os.path.join(directory, file_name))
        contents = r.read()
        r.close()

        sh = SecureHash()
        sh.update(contents)

        hash_format = '%08x'
        hash_id = hash_format % sh.digest()

        ut.load_kt(hash_id, contents)
        desired_kt_hash_tc_values[hash_id] = 10

    ut.log_core_tensors_to_json(os.path.join(directory, 'universal_knowledge_tensor.json'))

    print('loading Test BlockChain')
    bc = TestBlockChainCreator.generate(desired_kt_hash_tc_values,
                                        os.path.join(directory, 'truth_coin_block_chain.json'))

    ut.process_block_chain(bc)

    print('calculate all LifeScores')
    ut.calculate_life_score(test_phase1.root, directory)

    ut.log_results_to_json(os.path.join(directory, 'life_score_all_actions.json'))

    range_coord_mapper = RangeCoordMapper(ut.time_range)

    print('showing SingleActionViewer')
    single_score_viewer = SingleActionViewer(ut.action_TO_life_score_results)
    single_score_viewer.mainloop()

    print('showing MultiActionViewer')
    multi_score_viewer = MultiActionViewer(ut.action_TO_life_score_results)
    multi_score_viewer.mainloop()

    print('showing CoreViewer')
    core_viewer = CoreViewer(ut, range_coord_mapper)
    core_viewer.mainloop()


if __name__ == '__main__':
    main()
