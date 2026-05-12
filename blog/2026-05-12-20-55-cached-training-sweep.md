# Cached Training Sweep

`TICKET-0051` adds a bounded cached IDX training sweep to the normal e2e
manifest.

The new check creates a temporary train-split MNIST IDX cache with sixteen
deterministic samples, verifies the missing-cache error path, then reloads the
samples through the same cache resolver used by the opt-in cached command. The
pixel data carries a simple label-specific signal so the existing host training
loop has a stable target: in the direct check, loss moved from `2.302307` to
`1.901221` and accuracy from `0.12` to `0.38` over four epochs.

This still does not claim direct external runtime execution of the MNIST
classifier. It proves the command-facing cached-data route is exercised by the
default gate without network access or dependence on a user-local MNIST cache.
