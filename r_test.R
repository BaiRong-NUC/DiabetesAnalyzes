# 简单的 R 测试脚本：检查环境、常用包，并尝试读取 diabetes.csv
cat("=== R Environment Test ===\n")
cat("R version:")
print(R.version.string)

required <- c("ggplot2", "dplyr", "readr")
missing_pkgs <- required[!sapply(required, requireNamespace, quietly = TRUE)]
if (length(missing_pkgs) > 0) {
    cat("Missing packages:", paste(missing_pkgs, collapse = ", "), "\n")
    cat(
        "You can install them by running: install.packages(c(",
        paste0("'", missing_pkgs, "'", collapse = ","),
        "))\n"
    )
} else {
    cat("All required packages are installed.\n")
}

cat("Package availability:\n")
for (p in required) {
    cat(sprintf(" - %s: %s\n", p, requireNamespace(p, quietly = TRUE)))
}

csv_file <- "diabetes.csv"
if (file.exists(csv_file)) {
    cat("Found file:", csv_file, "\n")
    df <- tryCatch(
        read.csv(csv_file, stringsAsFactors = FALSE),
        error = function(e) e
    )
    if (inherits(df, "error")) {
        cat("Error reading CSV:", df$message, "\n")
    } else {
        cat("Rows:", nrow(df), "Cols:", ncol(df), "\n")
        cat("First 3 rows:\n")
        print(head(df, 3))
        cat("Summary (numeric columns):\n")
        numcols <- sapply(df, is.numeric)
        if (any(numcols)) {
            print(summary(df[, numcols, drop = FALSE]))
        } else {
            cat("No numeric columns found.\n")
        }
    }
} else {
    cat(csv_file, "not found in workspace.\n")
}

cat("=== Test complete ===\n")
