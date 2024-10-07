package org.NGramModel;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class CombineNGramCounts {

    public CombineNGramCounts() {
    }

    // Method to combine counts into a shared map
    public static void combineNGramCounts(Map<String, Integer> sharedMap, Map<String, Integer> mapToAdd) {
        mapToAdd.forEach((key, value) -> sharedMap.merge(key, value, Integer::sum));
    }

    // Get all file paths (only .csv files, no directories)
    public static List<Path> getAllFilePaths(String directoryPath) {
        try (Stream<Path> paths = Files.walk(Paths.get(directoryPath))) {
            return paths.filter(Files::isDirectory)       // Only regular files, not directories
                        .collect(Collectors.toList());
        } catch (IOException e) {
            e.printStackTrace();
            return Collections.emptyList();
        }
    }

    public static void main(String[] args) {
        String directoryPath = "final_dict/ngram_models";  // Path to the directory

        // Get list of all file paths (CSV files only)
        List<Path> filePaths = getAllFilePaths(directoryPath);

        // Use thread-safe maps to store combined counts
        ConcurrentMap<String, Integer> tempUni = new ConcurrentHashMap<>();
        ConcurrentMap<String, Integer> tempBi = new ConcurrentHashMap<>();
        ConcurrentMap<String, Integer> tempTri = new ConcurrentHashMap<>();

        // Determine the number of available processors
        int numProcessors = Runtime.getRuntime().availableProcessors();
        ExecutorService executorService = Executors.newFixedThreadPool(numProcessors);

        List<Future<Void>> futures = new ArrayList<>();
        NGramModel combinedModel = new NGramModel();  // This will hold the combined result

        // Process files in parallel
        for (Path filePath : filePaths) {
            futures.add(executorService.submit(() -> {
                NGramModel localModel = new NGramModel();
                localModel.loadModelFromCSV(filePath.toString());  // Load n-gram counts from CSV

                // // Combine the loaded counts with the master counts
                // combineNGramCounts(tempUni, localModel.getUnigramCounts());
                // combineNGramCounts(tempBi, localModel.getBigramCounts());
                // combineNGramCounts(tempTri, localModel.getTrigramCounts());

                return null;
            }));
        }

        // Wait for all futures to complete
        for (Future<Void> future : futures) {
            try {
                future.get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }

        // Shutdown the executor service
        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(60, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
        }

        // // Set the combined counts in the final model
        // combinedModel.setUnigramCounts(tempUni);
        // combinedModel.setBigramCounts(tempBi);
        // combinedModel.setTrigramCounts(tempTri);

        // Save the combined model to CSV
        combinedModel.saveModelToCSV("final_dict");
        System.out.println("Processing complete! Combined model saved to 'final_dict'.");
    }
}
