use tracing::instrument;

#[derive(Debug, Clone)]
pub struct Segment<A> {
    pub start: usize,
    pub end: usize,
    pub data: A,
}

impl<A> Segment<A> {
    pub fn new(start: usize, end: usize, data: A) -> Self {
        assert!(start < end, "Start must be less than end");
        Self { start, end, data }
    }

    pub fn size(&self) -> usize {
        self.end - self.start
    }

    pub fn overlaps(&self, other: &Segment<A>) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[instrument(skip_all, fields(n_segments = segments.len()))]
pub fn build_non_overlapping_partition<A>(segments: Vec<Segment<A>>) -> Vec<Vec<Segment<A>>> {
    if segments.is_empty() {
        return vec![];
    }

    // Create indexed segments and sort by size (largest first)
    let mut indexed_segments: Vec<(usize, Segment<A>)> = segments.into_iter().enumerate().collect();

    // Sort by size descending (larger segments first for better load balancing)
    indexed_segments.sort_by(|a, b| b.1.size().cmp(&a.1.size()));

    // Groups with their current load
    let mut groups: Vec<(Vec<Segment<A>>, usize)> = Vec::new();

    for (_, segment) in indexed_segments {
        let segment_size = segment.size();

        // Find the group with minimum load that doesn't have conflicts
        let mut best_group_idx = None;
        let mut min_load = usize::MAX;

        for (idx, (group, load)) in groups.iter().enumerate() {
            // Check if this segment conflicts with any in the group
            let has_conflict = group.iter().any(|s| s.overlaps(&segment));

            if !has_conflict && *load < min_load {
                min_load = *load;
                best_group_idx = Some(idx);
            }
        }

        match best_group_idx {
            Some(idx) => {
                // Add to existing group
                groups[idx].0.push(segment);
                groups[idx].1 += segment_size;
            }
            None => {
                // Create new group
                groups.push((vec![segment], segment_size));
            }
        }
    }

    // Extract just the segment groups (without loads)
    groups.into_iter().map(|(group, _)| group).collect()
}

// Helper function to analyze the partition quality
pub(crate) fn analyze_partition<A>(groups: &[Vec<Segment<A>>]) {
    for (i, group) in groups.iter().enumerate() {
        let total_size: usize = group.iter().map(|s| s.size()).sum();
        let max_size = group.iter().map(|s| s.size()).max().unwrap();
        let min_size = group.iter().map(|s| s.size()).min().unwrap();
        println!(
            "Group {}: segments = {}, total size = {}, max size = {}, min size = {}",
            i,
            group.len(),
            total_size,
            max_size,
            min_size,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn test_no_overlaps() {
        let segments = vec![
            Segment::new(0, 5, ()),
            Segment::new(5, 10, ()),
            Segment::new(10, 15, ()),
            Segment::new(15, 20, ()),
        ];

        let groups = build_non_overlapping_partition(segments);

        // Should all be in one group since no overlaps
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 4);
    }

    #[test]
    fn test_complete_overlap() {
        let segments = vec![
            Segment::new(0, 10, ()),
            Segment::new(0, 10, ()),
            Segment::new(0, 10, ()),
            Segment::new(0, 10, ()),
        ];

        let groups = build_non_overlapping_partition(segments);

        // Should be 4 groups since all overlap
        assert_eq!(groups.len(), 4);
        for group in &groups {
            assert_eq!(group.len(), 1);
        }
    }

    #[test]
    fn test_partial_overlaps() {
        let segments = vec![
            Segment::new(0, 10, ()),  // size 10
            Segment::new(5, 15, ()),  // size 10, overlaps with first
            Segment::new(12, 20, ()), // size 8, overlaps with second
            Segment::new(25, 30, ()), // size 5, no overlaps
            Segment::new(26, 35, ()), // size 9, overlaps with previous
        ];

        let groups = build_non_overlapping_partition(segments);
        analyze_partition(&groups);

        // Verify no overlaps within groups
        for group in &groups {
            for i in 0..group.len() {
                for j in (i + 1)..group.len() {
                    assert!(!group[i].overlaps(&group[j]));
                }
            }
        }
    }

    #[test]
    fn test_load_balancing() {
        let segments = vec![
            Segment::new(0, 20, ()),  // size 20
            Segment::new(25, 35, ()), // size 10
            Segment::new(40, 45, ()), // size 5
            Segment::new(50, 65, ()), // size 15
            Segment::new(70, 78, ()), // size 8
            Segment::new(80, 82, ()), // size 2
        ];

        let groups = build_non_overlapping_partition(segments);
        analyze_partition(&groups);

        // All should be in one group (no overlaps)
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn real_world_test() {
        let mut rng = StdRng::seed_from_u64(0);
        // Example usage
        let global_segment_size = 2000;
        let n_segments = 1000;
        let segments: Vec<Segment<()>> = (0..n_segments)
            .map(|_| {
                let start = rng.random_range(0..(global_segment_size - 10));
                let end = rng.random_range((start + 1)..=global_segment_size);
                Segment::new(start, end, ())
            })
            .collect();
        let time = std::time::Instant::now();
        let groups = build_non_overlapping_partition(segments);
        let duration = time.elapsed();
        println!("Partitioning took: {:?}", duration);
        analyze_partition(&groups);
    }
}
