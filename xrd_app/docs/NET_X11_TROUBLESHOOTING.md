# `/net` and SSH/X11 Crash Notes

## Observed Failure

When `xrd-app gui` or `xrf-app gui` is run through SSH while the project and raw
HDF5 files are accessed through `/net`, a stalled network filesystem can coincide
with the GUI disappearing or the X11 connection breaking.

A later attempt to launch the GUI may fail immediately with output similar to:

```text
qt.qpa.xcb: could not connect to display :0
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" even though it was found.
This application failed to start because no Qt platform plugin could be initialized.
```

In this sequence, the `xcb` message is usually a secondary symptom: the first
failure has already left the X display or forwarding session unavailable. It does
not normally mean that Qt or the `xcb` plugin must be reinstalled.

## Recovery

Try these in order:

1. Stop the failed `xrd-app` or `xrf-app` process.
2. Disconnect and reconnect SSH with X11 forwarding, preferably `ssh -Y host` or
   otherwise `ssh -X host`.
3. Confirm that the new shell has a forwarded display such as
   `DISPLAY=localhost:10.0`. Do not manually set `DISPLAY=:0` for an SSH-forwarded
   session.
4. If the display session remains broken, log out and back in or restart the
   window-manager/display-manager session.
5. If those steps do not recover the session, restart the whole computer. Killing
   the window-manager process was also observed to recover the affected machine.

The launch commands perform a display preflight before creating Qt. If the second
launch cannot reach the display, they now report an X11 connection problem instead
of allowing Qt to terminate with the misleading plugin-reinstall message.

## Application Mitigation

View/Label does not persist a detector-image cache. When it must use loose raw
frames, detector HDF5 reads run in a separate child-process broker:

- Qt does not open the raw `/net` HDF5 files directly.
- Completed detector images are returned to the GUI and retained only in the
  bounded in-memory image cache.
- If a raw read fails, the last good image remains visible and selecting the bin
  again retries the request.
- If the grid mapping is missing, **Prepare raw view** builds only the small
  frame-to-grid index; it does not copy detector images.

This isolates raw detector reads, but the project configuration and small metadata
files still live under the project path. A completely unresponsive `/net` mount
can therefore still affect project startup outside the raw-image broker.

Eight seconds after either GUI command starts, the terminal prints a reminder that
a `/net` stall may have broken X11 and gives the observed recovery options.
