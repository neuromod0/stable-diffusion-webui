from __future__ import annotations

import configparser
import os
import threading
import re

from components import shared, scripts
from utils import errors,cache
from components.gitpython_hack import Repo
from components.paths_internal import extensions_dir, extensions_builtin_dir, script_path  # noqa: F401


os.makedirs(extensions_dir, exist_ok=True)


def active():
    if shared.cmd_opts.disable_all_extensions or shared.opts.disable_all_extensions == "all":
        return []
    elif shared.cmd_opts.disable_extra_extensions or shared.opts.disable_all_extensions == "extra":
        return [x for x in extensions if x.enabled and x.is_builtin]
    else:
        return [x for x in extensions if x.enabled]


class ExtensionMetadata:
    filename = "metadata.ini"
    config: configparser.ConfigParser
    canonical_name: str
    requires: list

    def __init__(self, path, canonical_name):
        self.config = configparser.ConfigParser()

        filepath = os.path.join(path, self.filename)
        if os.path.isfile(filepath):
            try:
                self.config.read(filepath)
            except Exception:
                errors.report(f"Error reading {self.filename} for extension {canonical_name}.", exc_info=True)

        self.canonical_name = self.config.get("Extension", "Name", fallback=canonical_name)
        self.canonical_name = canonical_name.lower().strip()

        self.requires = self.get_script_requirements("Requires", "Extension")

    def get_script_requirements(self, field, section, extra_section=None):
        """reads a list of requirements from the config; field is the name of the field in the ini file,
        like Requires or Before, and section is the name of the [section] in the ini file; additionally,
        reads more requirements from [extra_section] if specified."""

        x = self.config.get(section, field, fallback='')

        if extra_section:
            x = x + ', ' + self.config.get(extra_section, field, fallback='')

        return self.parse_list(x.lower())

    def parse_list(self, text):
        """converts a line from config ("ext1 ext2, ext3  ") into a python list (["ext1", "ext2", "ext3"])"""

        if not text:
            return []

        # both "," and " " are accepted as separator
        return [x for x in re.split(r"[,\s]+", text.strip()) if x]


class Extension:
    lock = threading.Lock()
    cached_fields = ['remote', 'commit_date', 'branch', 'commit_hash', 'version']
    metadata: ExtensionMetadata

    def __init__(self, name, path, enabled=True, is_builtin=False, metadata=None):
        self.name = name
        self.path = path
        self.enabled = enabled
        self.status = ''
        self.can_update = False
        self.is_builtin = is_builtin
        self.commit_hash = ''
        self.commit_date = None
        self.version = ''
        self.branch = None
        self.remote = None
        self.have_info_from_repo = False
        self.metadata = metadata if metadata else ExtensionMetadata(self.path, name.lower())
        self.canonical_name = metadata.canonical_name

    def to_dict(self):
        return {x: getattr(self, x) for x in self.cached_fields}

    def from_dict(self, d):
        for field in self.cached_fields:
            setattr(self, field, d[field])

    def read_info_from_repo(self):
        if self.is_builtin or self.have_info_from_repo:
            return

        def read_from_repo():
            with self.lock:
                if self.have_info_from_repo:
                    return

                self.do_read_info_from_repo()

                return self.to_dict()

        try:
            d = cache.cached_data_for_file('extensions-git', self.name, os.path.join(self.path, ".git"), read_from_repo)
            self.from_dict(d)
        except FileNotFoundError:
            pass
        self.status = 'unknown' if self.status == '' else self.status

    def do_read_info_from_repo(self):
        repo = None
        try:
            if os.path.exists(os.path.join(self.path, ".git")):
                repo = Repo(self.path)
        except Exception:
            errors.report(f"Error reading github repository info from {self.path}", exc_info=True)

        if repo is None or repo.bare:
            self.remote = None
        else:
            try:
                self.remote = next(repo.remote().urls, None)
                commit = repo.head.commit
                self.commit_date = commit.committed_date
                if repo.active_branch:
                    self.branch = repo.active_branch.name
                self.commit_hash = commit.hexsha
                self.version = self.commit_hash[:8]

            except Exception:
                errors.report(f"Failed reading extension data from Git repository ({self.name})", exc_info=True)
                self.remote = None

        self.have_info_from_repo = True

    def list_files(self, subdir, extension):
        dirpath = os.path.join(self.path, subdir)
        if not os.path.isdir(dirpath):
            return []

        res = []
        for filename in sorted(os.listdir(dirpath)):
            res.append(scripts.ScriptFile(self.path, filename, os.path.join(dirpath, filename)))

        res = [x for x in res if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

        return res

    def check_updates(self):
        repo = Repo(self.path)
        for fetch in repo.remote().fetch(dry_run=True):
            if fetch.flags != fetch.HEAD_UPTODATE:
                self.can_update = True
                self.status = "new commits"
                return

        try:
            origin = repo.rev_parse('origin')
            if repo.head.commit != origin:
                self.can_update = True
                self.status = "behind HEAD"
                return
        except Exception:
            self.can_update = False
            self.status = "unknown (remote error)"
            return

        self.can_update = False
        self.status = "latest"

    def fetch_and_reset_hard(self, commit='origin'):
        repo = Repo(self.path)
        # Fix: `error: Your local changes to the following files would be overwritten by merge`,
        # because WSL2 Docker set 755 file permissions instead of 644, this results to the error.
        repo.git.fetch(all=True)
        repo.git.reset(commit, hard=True)
        self.have_info_from_repo = False


def list_extensions():
    extensions.clear()

    if shared.cmd_opts.disable_all_extensions:
        print("*** \"--disable-all-extensions\" arg was used, will not load any extensions ***")
    elif shared.opts.disable_all_extensions == "all":
        print("*** \"Disable all extensions\" option was set, will not load any extensions ***")
    elif shared.cmd_opts.disable_extra_extensions:
        print("*** \"--disable-extra-extensions\" arg was used, will only load built-in extensions ***")
    elif shared.opts.disable_all_extensions == "extra":
        print("*** \"Disable all extensions\" option was set, will only load built-in extensions ***")

    loaded_extensions = {}

    # scan through extensions directory and load metadata
    for dirname in [extensions_builtin_dir, extensions_dir]:
        if not os.path.isdir(dirname):
            continue

        for extension_dirname in sorted(os.listdir(dirname)):
            path = os.path.join(dirname, extension_dirname)
            if not os.path.isdir(path):
                continue

            canonical_name = extension_dirname
            metadata = ExtensionMetadata(path, canonical_name)

            # check for duplicated canonical names
            already_loaded_extension = loaded_extensions.get(metadata.canonical_name)
            if already_loaded_extension is not None:
                errors.report(f'Duplicate canonical name "{canonical_name}" found in extensions "{extension_dirname}" and "{already_loaded_extension.name}". Former will be discarded.', exc_info=False)
                continue

            is_builtin = dirname == extensions_builtin_dir
            extension = Extension(name=extension_dirname, path=path, enabled=extension_dirname not in shared.opts.disabled_extensions, is_builtin=is_builtin, metadata=metadata)
            extensions.append(extension)
            loaded_extensions[canonical_name] = extension

    # check for requirements
    for extension in extensions:
        for req in extension.metadata.requires:
            required_extension = loaded_extensions.get(req)
            if required_extension is None:
                errors.report(f'Extension "{extension.name}" requires "{req}" which is not installed.', exc_info=False)
                continue

            if not extension.enabled:
                errors.report(f'Extension "{extension.name}" requires "{required_extension.name}" which is disabled.', exc_info=False)
                continue


extensions: list[Extension] = []
