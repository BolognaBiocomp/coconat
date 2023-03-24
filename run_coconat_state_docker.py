#!/usr/bin/env python
import os
import pathlib
import signal
from typing import Tuple

from absl import app
from absl import flags
from absl import logging
import docker
from docker import types

flags.DEFINE_string('fasta_file', None, 'Path to FASTA file')
flags.DEFINE_string('seg_file', None, 'Path to CC segment file')
flags.DEFINE_string('output_file', None, 'Output file prefix')
flags.DEFINE_string('plm_dir', None, 'CoCoNat pLM dir')
flags.DEFINE_string('docker_image_name', 'coconat:1.0',
    'Name of the CoCoNat Docker image.')
flags.DEFINE_string('docker_user', f'{os.geteuid()}:{os.getegid()}',
    'UID:GID with which to run the Docker container.')

FLAGS = flags.FLAGS

_ROOT_MOUNT_DIRECTORY = '/mnt/'

def _create_mount(mount_name: str, path: str) -> Tuple[types.Mount, str]:
  """Create a mount point for each file and directory used by the model."""
  path = pathlib.Path(path).absolute()
  target_path = pathlib.Path(_ROOT_MOUNT_DIRECTORY, mount_name)
  print(path)
  if path.is_dir():
    source_path = path
    mounted_path = target_path
  else:
    source_path = path.parent
    mounted_path = pathlib.Path(target_path, path.name)
  if not source_path.exists():
    raise ValueError(f'Failed to find source directory "{source_path}" to '
                     'mount in Docker container.')
  logging.info('Mounting %s -> %s', source_path, target_path)
  mount = types.Mount(target=str(target_path), source=str(source_path),
                      type='bind', read_only=True)
  return mount, str(mounted_path)

def main(argv):
  if len(argv) > 1:
      raise app.UsageError('Too many command-line arguments.')

  # You can individually override the following paths if you have placed the
  # data in locations other than the FLAGS.data_dir.

  command_args = []

  mnt_plm = "/mnt/plms"
  mnt_input = "/mnt/input"
  mnt_output = "/mnt/output"

  source_plm_dir = str(pathlib.Path(FLAGS.plm_dir).absolute())
  source_input_dir = str(pathlib.Path(os.path.dirname(FLAGS.fasta_file)).absolute())
  source_output_dir = str(pathlib.Path(os.path.dirname(FLAGS.output_file)).absolute())

  if source_input_dir != source_output_dir:
      target_fasta_file = os.path.join(mnt_input, os.path.basename(FLAGS.fasta_file))
      target_seg_file = os.path.join(mnt_input, os.path.basename(FLAGS.seg_file))
      target_out_file = os.path.join(mnt_output, os.path.basename(FLAGS.output_file))
      volume_cfg = {source_plm_dir: {"bind": mnt_plm, "mode":'ro'},
                    source_input_dir: {"bind": mnt_input, "mode":'ro'},
                    source_output_dir: {"bind": mnt_output, "mode":"rw"}}
  else:
      target_fasta_file = os.path.join(mnt_input, os.path.basename(FLAGS.fasta_file))
      target_seg_file = os.path.join(mnt_input, os.path.basename(FLAGS.seg_file))
      target_out_file = os.path.join(mnt_input, os.path.basename(FLAGS.output_file))
      mnt_output = mnt_input
      volume_cfg = {source_plm_dir: {"bind": mnt_plm, "mode":'ro'},
                    source_input_dir: {"bind": mnt_input, "mode":'rw'}}

  print("Mounting %s on %s" % (source_plm_dir, mnt_plm))
  print("Mounting %s on %s" % (source_input_dir, mnt_input))
  print("Mounting %s on %s" % (source_output_dir, mnt_output))

  command_args.extend([
         'state',
         '-f',
         f'{target_fasta_file}',
         '-s',
         f'{target_seg_file}',
         '-o',
         f'{target_out_file}'
         ])

  client = docker.from_env()
  container = client.containers.run(
      image=FLAGS.docker_image_name,
      command=command_args,
      device_requests=None,
      remove=True,
      detach=True,
      volumes = volume_cfg,
      user=FLAGS.docker_user,
      environment={

      })
  # Add signal handler to ensure CTRL+C also stops the running container.
  signal.signal(signal.SIGINT,
                lambda unused_sig, unused_frame: container.kill())

  for line in container.logs(stream=True):
    logging.info(line.strip().decode('utf-8'))


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_file',
      'seg_file',
      'plm_dir',
      'output_file',
  ])
  app.run(main)
