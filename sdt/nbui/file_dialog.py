# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import ipywidgets
import traitlets


class FileDialog(ipywidgets.VBox):
    """File selection dialog"""
    cur_dir = traitlets.Any()
    """Current directory"""
    selected = traitlets.List()
    """List of selected files"""

    def __init__(self, cur_dir=Path(), *args, **kwargs):
        """Parameters
        ----------
        cur_dir : Path
            Current directory
        *args, **kwargs
            Passed to super class constructor
        """
        self._start_button = ipywidgets.Button(description="Select files…")
        self._start_button.on_click(self._show_dialog)
        self._dir_button_box = ipywidgets.HBox()
        self._dir_list = ipywidgets.Select()
        self._file_list = ipywidgets.SelectMultiple()
        self._ok_button = ipywidgets.Button(description="OK")
        self._ok_button.on_click(self._ok_pressed)
        self._cancel_button = ipywidgets.Button(description="Cancel")
        self._cancel_button.on_click(self._cancel_pressed)
        self._dialog_box = ipywidgets.VBox([
                ipywidgets.Label(value="Current folder"),
                self._dir_button_box,
                ipywidgets.HBox([
                    ipywidgets.VBox([
                        ipywidgets.Label(value="Choose subfolder"),
                        self._dir_list
                    ]),
                    ipywidgets.VBox([ipywidgets.Label(value="Select files"),
                                    self._file_list])
                ]),
                ipywidgets.HBox([self._ok_button, self._cancel_button])
            ], layout=ipywidgets.Layout(display="none"))

        super().__init__([self._start_button, self._dialog_box],
                         *args, **kwargs)

        self._ignore_dir_selected = False
        self._dir_list.observe(self._dir_selected, "value")
        self.cur_dir = cur_dir

    @traitlets.validate("cur_dir")
    def _validate_cur_dir(self, proposal):
        return Path(proposal["value"]).resolve()

    @traitlets.observe("cur_dir")
    def _cur_dir_changed(self, change=None):
        self._set_tmp_cwd(self.cur_dir)

    def _set_tmp_cwd(self, tmp_cwd):
        """Set temporay cur dir and update widgets

        Parameters
        ----------
        tmp_cwd : Path
            New temp directory
        """
        self._tmp_cwd = tmp_cwd

        files = []
        dirs = []

        for p in self._tmp_cwd.iterdir():
            if p.name.startswith("."):
                continue
            if p.is_dir():
                dirs.append(str(p.relative_to(self._tmp_cwd)))
            else:
                files.append(str(p.relative_to(self._tmp_cwd)))

        self._file_list.options = sorted(files)
        try:
            self._ignore_dir_selected = True
            self._dir_list.options = sorted(dirs)
            self._dir_list.value = None
        except Exception:
            raise
        finally:
            self._ignore_dir_selected = False

        btns = []
        for p in self._tmp_cwd.parts:
            b = ipywidgets.Button(description=p,
                                  layout=ipywidgets.Layout(width="auto"))
            b.on_click(self._dir_button_clicked)
            btns.append(b)
        self._dir_button_box.children = btns

    def _dir_button_clicked(self, button):
        """Called when one of the "current folder" buttons is clicked"""
        bbc = self._dir_button_box.children
        idx = bbc.index(button)
        if idx == len(bbc) - 1:
            # Last button ==> current dir, do nothing
            return
        self._set_tmp_cwd(self._tmp_cwd.parents[len(bbc) - idx - 2])

    def _show_dialog(self, button=None):
        """Show the dialog"""
        self._start_button.layout = ipywidgets.Layout(display="none")
        self._dialog_box.layout = ipywidgets.Layout(display="inline")

    def _dir_selected(self, change=None):
        """Called when a subdir was selected from the list"""
        if self._ignore_dir_selected:
            return
        self._set_tmp_cwd(self._tmp_cwd / self._dir_list.value)

    def _ok_pressed(self, button=None):
        """OK button pressed after selection"""
        self._start_button.layout = ipywidgets.Layout(display="inline")
        self._dialog_box.layout = ipywidgets.Layout(display="none")

        self.selected = [self._tmp_cwd / s for s in self._file_list.value]
        self.cur_dir = self._tmp_cwd

    def _cancel_pressed(self, button=None):
        """Cancel button pressed"""
        self._start_button.layout = ipywidgets.Layout(display="inline")
        self._dialog_box.layout = ipywidgets.Layout(display="none")
        self._set_tmp_cwd(self.cur_dir)
