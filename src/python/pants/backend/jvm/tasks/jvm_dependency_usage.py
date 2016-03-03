# coding=utf-8
# Copyright 2015 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import (absolute_import, division, generators, nested_scopes, print_function,
                        unicode_literals, with_statement)

import json
import os
import sys
from collections import defaultdict, namedtuple

from pants.backend.jvm.targets.jar_library import JarLibrary
from pants.backend.jvm.tasks.classpath_util import ClasspathUtil
from pants.backend.jvm.tasks.ivy_task_mixin import IvyResolveFingerprintStrategy
from pants.backend.jvm.tasks.jvm_dependency_analyzer import JvmDependencyAnalyzer
from pants.base.build_environment import get_buildroot
from pants.base.exceptions import TaskError
from pants.build_graph.resources import Resources
from pants.build_graph.target import Target
from pants.invalidation.cache_manager import VersionedTargetSet
from pants.util.dirutil import fast_relpath, safe_mkdir
from pants.util.fileutil import create_size_estimators


class JvmDependencyUsage(JvmDependencyAnalyzer):
  """Determines the dependency usage ratios of targets.

  Analyzes the relationship between the products a target T produces vs. the products
  which T's dependents actually require (this is done by observing analysis files).
  If the ratio of required products to available products is low, then this is a sign
  that target T isn't factored well.

  A graph is formed from these results, where each node of the graph is a target, and
  each edge is a product usage ratio between a target and its dependency. The nodes
  also contain additional information to guide refactoring -- for example, the estimated
  job size of each target, which indicates the impact a poorly factored target has on
  the build times. (see DependencyUsageGraph->to_json)

  The graph is either summarized for local analysis or outputted as a JSON file for
  aggregation and analysis on a larger scale.
  """

  size_estimators = create_size_estimators()

  @classmethod
  def register_options(cls, register):
    super(JvmDependencyUsage, cls).register_options(register)
    register('--internal-only', default=True, action='store_true',
             help='Specifies that only internal dependencies should be included in the graph '
                  'output (no external jars).')
    register('--summary', default=True, action='store_true',
             help='When set, outputs a summary of the "worst" dependencies; otherwise, '
                  'outputs a JSON report.')
    register('--size-estimator',
             choices=list(cls.size_estimators.keys()), default='filesize',
             help='The method of target size estimation.')
    register('--transitive', default=True, action='store_true',
             help='Score all targets in the build graph transitively.')
    register('--output-file', type=str,
             help='Output destination. When unset, outputs to <stdout>.')
    register('--only-cached', action='store_true', help='Use only cached value, fail otherwise.')

  @classmethod
  def prepare(cls, options, round_manager):
    super(JvmDependencyUsage, cls).prepare(options, round_manager)
    if not options.only_cached:
      round_manager.require_data('classes_by_source')
      round_manager.require_data('runtime_classpath')
      round_manager.require_data('product_deps_by_src')

  @classmethod
  def implementation_version(cls):
    return super(JvmDependencyUsage, cls).implementation_version() + [('JvmDependencyUsage', 1)]

  def check_artifact_cache_for(self, invalidation_check):
    # Jvm dependency usage depends on ivy resolve which is global, so we are using the same strategy as in ivy here.
    return [self.task_vts(invalidation_check)]

  def task_vts(self, invalidation_check):
    return VersionedTargetSet.from_versioned_targets(invalidation_check.all_vts)

  @classmethod
  def skip(cls, options):
    """This task is always explicitly requested."""
    return False

  def execute(self):
    targets = (self.context.targets() if self.get_options().transitive
               else self.context.target_roots)
    graph = self.get_dep_usage_graph(targets, get_buildroot())

    output_file = self.get_options().output_file
    if output_file:
      self.context.log.info('Writing dependency usage to {}'.format(output_file))
      with open(output_file, 'w') as fh:
        self._render(graph, fh)
    else:
      sys.stdout.write(b'\n')
      self._render(graph, sys.stdout)

  def _render(self, graph, fh):
    chunks = graph.to_summary() if self.get_options().summary else [graph.to_json()]
    for chunk in chunks:
      fh.write(chunk)
    fh.flush()

  def _resolve_aliases(self, target):
    """Recursively resolve `target` aliases."""
    for declared in target.dependencies:
      if type(declared) == Target:
        for r in self._resolve_aliases(declared):
          yield r
      else:
        yield declared

  def _is_declared_dep(self, target, dep):
    """Returns true if the given dep target should be considered a declared dep of target."""
    return dep in self._resolve_aliases(target)

  def _select(self, target):
    if self.get_options().internal_only and isinstance(target, JarLibrary):
      return False
    elif isinstance(target, Resources) or type(target) == Target:
      return False
    else:
      return True

  def _normalize_product_dep(self, buildroot, classes_by_source, dep):
    """Normalizes the given product dep from the given dep into a set of classfiles.

    Product deps arrive as sources, jars, and classfiles: this method normalizes them to classfiles.

    TODO: This normalization should happen in the super class.
    """
    if dep.endswith(".jar"):
      # TODO: post sbt/zinc jar output patch, binary deps will be reported directly as classfiles
      return set()
    elif dep.endswith(".class"):
      return set([dep])
    else:
      # assume a source file and convert to classfiles
      rel_src = fast_relpath(dep, buildroot)
      return set(p for _, paths in classes_by_source[rel_src].rel_paths() for p in paths)

  def _count_products(self, classpath_products, target):
    contents = ClasspathUtil.classpath_contents((target,), classpath_products)
    # Generators don't implement len.
    return sum(1 for _ in contents)

  def get_dep_usage_graph(self, targets, buildroot):
    fingerprint_strategy = IvyResolveFingerprintStrategy(('default',))
    with self.invalidated(targets,
                          fingerprint_strategy=fingerprint_strategy) as invalidation_check:
      if not invalidation_check.all_vts:
        return DependencyUsageGraph({})

      vts = self.task_vts(invalidation_check)
      hash = vts.cache_key.hash

      graph_json = os.path.join(self.workdir, 'graph_{}.json'.format(hash))
      if not os.path.exists(graph_json):
        if self.get_options().only_cached:
          raise TaskError("only_cached options was passed, but usages graph wasn't in buildcache.")

        graph = self.create_dep_usage_graph(targets, buildroot)
        safe_mkdir(self.workdir)
        with open(graph_json, mode='w') as fp:
          fp.write(graph.to_json())
        if self.artifact_cache_writes_enabled():
          self.update_artifact_cache([(vts, [graph_json])])
        return graph
      else:
        with open(graph_json) as fp:
          return DependencyUsageGraph.from_json(fp.read())

  def create_nodes_with_costs(self, targets):
    cost_cache = {}
    trans_cost_cache = {}
    size_estimator = self.size_estimators[self.get_options().size_estimator]

    def cost(target):
      if target not in cost_cache:
        cost_cache[target] = size_estimator(target.sources_relative_to_buildroot())
      return cost_cache[target]

    def trans_cost(target):
      if target not in trans_cost_cache:
        dep_sum = sum(trans_cost(dep) for dep in target.dependencies)
        trans_cost_cache[target] = cost(target) + dep_sum
      return trans_cost_cache[target]

    nodes = dict()
    queue = set(targets)
    while queue:
      target = queue.pop()
      concrete_target = target.concrete_derived_from
      spec = concrete_target.address.spec
      if spec not in nodes:
        nodes[spec] = Node(spec, cost(concrete_target), trans_cost(concrete_target))
        queue.update(target.dependencies)
        queue.update(concrete_target.dependencies)

    return nodes

  def create_dep_usage_graph(self, targets, buildroot):
    """Creates a graph of concrete targets, with their sum of products and dependencies.

    Synthetic targets contribute products and dependencies to their concrete target.
    """

    # Initialize all Nodes.
    classes_by_source = self.context.products.get_data('classes_by_source')
    runtime_classpath = self.context.products.get_data('runtime_classpath')
    product_deps_by_src = self.context.products.get_data('product_deps_by_src')

    selected_targets = [target for target in targets if self._select(target)]
    nodes = self.create_nodes_with_costs(selected_targets)

    for target in selected_targets:
      # Create or extend a Node for the concrete version of this target.
      concrete_target = target.concrete_derived_from
      products_total = self._count_products(runtime_classpath, target)
      node = nodes.get(concrete_target.address.spec)
      node.add_derivation(target.address.spec, products_total)

      # Record declared Edges.
      for dep_tgt in self._resolve_aliases(target):
        derived_from = dep_tgt.concrete_derived_from
        if self._select(derived_from):
          node.add_edge(Edge(is_declared=True, products_used=set()), derived_from.address.spec)

      # Record the used products and undeclared Edges for this target. Note that some of
      # these may be self edges, which are considered later.
      target_product_deps_by_src = product_deps_by_src.get(target, dict())
      for src in target.sources_relative_to_buildroot():
        for product_dep in target_product_deps_by_src.get(os.path.join(buildroot, src), []):
          for dep_tgt in self.targets_by_file.get(product_dep, []):
            derived_from = dep_tgt.concrete_derived_from
            if not self._select(derived_from):
              continue
            is_declared = self._is_declared_dep(target, dep_tgt)
            normalized_deps = self._normalize_product_dep(buildroot, classes_by_source, product_dep)
            node.add_edge(Edge(is_declared=is_declared, products_used=normalized_deps), derived_from.address.spec)

    return DependencyUsageGraph(nodes)


class Node(object):
  def __init__(self, concrete_target, cost, trans_cost):
    self.concrete_target = concrete_target
    self.cost = cost
    self.trans_cost = trans_cost
    self.products_total = 0
    self.derivations = set()
    # Dict mapping concrete dependency targets to an Edge object.
    self.dep_edges = defaultdict(Edge)

  def add_derivation(self, derived_target, derived_products):
    self.derivations.add(derived_target)
    self.products_total += derived_products

  def add_edge(self, edge, dest):
    self.dep_edges[dest] += edge


class Edge(object):
  """Record a set of used products, and a boolean indicating that a depedency edge was declared."""

  def __init__(self, is_declared=False, products_used=None, loaded=False):
    self.products_used = products_used or set()
    self.is_declared = is_declared
    self.synthetic = loaded

  def __iadd__(self, that):
    if self.synthetic:
      raise Exception()
    self.products_used |= that.products_used
    self.is_declared |= that.is_declared
    return self


class DependencyUsageGraph(object):

  def __init__(self, nodes):
    self._nodes = nodes

  def _edge_type(self, target, edge, dep):
    if target == dep:
      return 'self'
    elif edge.is_declared:
      return 'declared'
    else:
      return 'undeclared'

  def _used_ratio(self, dep_tgt, edge):
    dep_tgt_products_total = max(self._nodes[dep_tgt].products_total if dep_tgt in self._nodes else 1, 1)
    return len(edge.products_used) / dep_tgt_products_total

  def to_summary(self):
    """Outputs summarized dependencies ordered by a combination of max usage and cost."""

    # Aggregate inbound edges by their maximum product usage ratio.
    max_target_usage = defaultdict(lambda: 0.0)
    for target, node in self._nodes.items():
      for dep_target, edge in node.dep_edges.items():
        if target == dep_target:
          continue
        used_ratio = self._used_ratio(dep_target, edge)
        max_target_usage[dep_target] = max(max_target_usage[dep_target], used_ratio)

    # Calculate a score for each.
    Score = namedtuple('Score', ('badness', 'max_usage', 'cost_transitive', 'target'))
    scores = []
    for target, max_usage in max_target_usage.items():
      if self._nodes[target].products_total == 0:
        continue

      cost_transitive = self._nodes[target].trans_cost
      score = int(cost_transitive / (max_usage if max_usage > 0.0 else 1.0))
      scores.append(Score(score, max_usage, cost_transitive, target))

    # Output in order by score.
    yield '[\n'
    first = True
    for score in sorted(scores, key=lambda s: s.badness):
      yield '{}  {}'.format('' if first else ',\n', json.dumps(score._asdict()))
      first = False
    yield '\n]\n'

  def to_json(self):
    """Outputs the entire graph."""
    res_dict = {}
    def gen_dep_edge(node, edge, dep_tgt):
      return {
        'target': dep_tgt,
        'dependency_type': self._edge_type(node.concrete_target, edge, dep_tgt),
        'products_used': len(edge.products_used),
        'products_used_ratio': self._used_ratio(dep_tgt, edge),
      }
    for node in self._nodes.values():
      res_dict[node.concrete_target] = {
          'cost': node.cost,
          'cost_transitive': node.trans_cost,
          'products_total': node.products_total,
          'derivations': list(node.derivations),
          'dependencies': [gen_dep_edge(node, edge, dep_tgt) for dep_tgt, edge in node.dep_edges.items()],
        }
    return json.dumps(res_dict, indent=2, sort_keys=True)

  @staticmethod
  def from_json(json_str):
    def edge_from_dict(dep):
      products_used = [str(i) for i in range(0, dep['products_used'])]
      return Edge(dep['dependency_type'] == 'self' or dep['dependency_type'] == 'declared',
                  products_used, loaded=True), dep['target']

    def node_from_dict(target, node_dict):
      node = Node(target, node_dict['cost'], node_dict['cost_transitive'])
      node.products_total = node_dict['products_total']
      node.derivations.update(node_dict['derivations'])
      for dep in node_dict['dependencies']:
        edge, dest = edge_from_dict(dep)
        node.dep_edges[dest] = edge
      return node

    original_dict = json.loads(json_str)
    nodes = {}
    for target, node_dict in original_dict.items():
      nodes[target] = node_from_dict(target, node_dict)

    return DependencyUsageGraph(nodes)
