library(ggplot2)

# Select the updated coherence_scores.csv in your assignment_5 folder.
csv_path <- file.choose()
scores <- read.csv("C:/Users/we203055/OneDrive - University of Central Florida/Desktop/assignment 5 1105/MAN7916/assignments/submissions/assignment_5/coherence_scores.csv")


# Find the topic count with the highest baseline coherence.
best <- scores[which.max(scores$coherence), ]
print(best)

# Plot all nine baseline models and highlight the highest score.
p <- ggplot(scores, aes(x = topics, y = coherence)) +
  geom_line(color = "steelblue") +
  geom_point(color = "steelblue", size = 3) +
  geom_point(data = best, color = "firebrick", size = 4) +
  scale_x_continuous(breaks = 2:10) +
  labs(
    title = "Baseline Coherence by Number of Topics",
    x = "Number of topics",
    y = "Coherence (c_v)"
  ) +
  theme_minimal()

print(p)

# Save the JPG in the same folder as the CSV.
jpg_path <- file.path(dirname(csv_path), "coherence_scores.jpg")
ggsave(jpg_path, plot = p, width = 7, height = 4.5, dpi = 300)