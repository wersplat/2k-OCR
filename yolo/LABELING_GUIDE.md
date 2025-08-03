# NBA 2K Box Score Labeling Guide

This guide provides instructions for labeling NBA 2K box score screenshots for YOLOv8 object detection training.

## Classes to Label

Label each stat block with the appropriate class:

1. `player_name` - Player name (left-most column)
2. `grade` - Player grade (A+, B-, etc.)
3. `points` - Points scored (PTS)
4. `rebounds` - Rebounds (REB)
5. `assists` - Assists (AST)
6. `steals` - Steals (STL)
7. `blocks` - Blocks (BLK)
8. `fouls` - Fouls (FOUL)
9. `turnovers` - Turnovers (TO)
10. `fgm_fga` - Field goals made/attempted (FGM/FGA)
11. `3pm_3pa` - Three-pointers made/attempted (3PM/3PA)
12. `ftm_fta` - Free throws made/attempted (FTM/FTA)

## Labeling Instructions

1. **Bounding Box Guidelines**:
   - Draw tight bounding boxes around each stat element
   - Include the entire text and a small margin (~2-5 pixels)
   - Be consistent with margins across all labels
   - Ensure boxes don't overlap

2. **Labeling Workflow**:
   - Label all 12 stats for each player row (10 players total)
   - Work from top to bottom (Team 1's 5 players, then Team 2's 5 players)
   - For each player row, label from left to right (name → grade → points → etc.)
   - Maintain consistent naming conventions

3. **Special Cases**:
   - For empty stats, still label the region where the stat would appear
   - For stats with fractions (e.g., "5/7"), include the entire fraction in one bounding box
   - If text is partially obscured, label the visible portion

## Export Format

When exporting labeled data:
1. Use YOLO format (txt files with normalized coordinates)
2. Organize in the following structure:
   ```
   data/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
   ```
3. Split data approximately 70% train, 20% validation, 10% test

## Example Coordinates

Based on the existing fixed regions in the original code, here are approximate coordinates for reference:

| Stat Type | X (approx) | Width (approx) |
|-----------|------------|----------------|
| name      | 1219       | 485            |
| grade     | 1699       | 105            |
| points    | 1839       | 135            |
| rebounds  | 1996       | 135            |
| assists   | 2142       | 135            |
| steals    | 2294       | 135            |
| blocks    | 2443       | 135            |
| fouls     | 2587       | 135            |
| turnovers | 2732       | 135            |
| fgm_fga   | 2884       | 205            |
| 3pm_3pa   | 3117       | 205            |
| ftm_fta   | 3318       | 205            |

Y-coordinates for player rows: 520, 602, 683, 765, 843, 1148, 1233, 1318, 1398, 1479

Note: These are reference values from the fixed cropping logic. Your actual bounding boxes should be based on the visual appearance of each stat in the screenshots.
