# Structure: the radial distribution g(r) of the teacher's sampling run (the frames of labeled_path)
# and of the student's run (student_frames_path), over all pairs, with the same bins.
# Gives: teacher_rdf, student_rdf.
# Names from earlier: helpers, labeled_path, student_frames_path, device.
from nvalchemi.training.distillation.evaluation import radial_distribution

r_max = 6.0      # Å
num_bins = 60    # bins of 0.1 Å


def frames_of(path):
    """Every frame of a Zarr store, as one Batch on the device."""
    _, batch = helpers.read_store(path)
    return batch.to(device)


teacher_rdf = radial_distribution(frames_of(labeled_path), r_max=r_max, num_bins=num_bins)
student_rdf = radial_distribution(frames_of(student_frames_path), r_max=r_max, num_bins=num_bins)
