"""Plugin discovery.

A plugin is a package under ``fovea.plugins`` exposing a module-level ``PLUGIN`` object
(see :class:`fovea.plugin_api.Plugin`). Its router is mounted at ``/api/plugins/<id>`` and
its ``static_dir`` at ``/plugins/<id>/``. The frontend imports ``entry`` and calls
``default(fovea)`` so the plugin can register views, tabs and commands.
"""
