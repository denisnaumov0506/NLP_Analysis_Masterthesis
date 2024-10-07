package org.NGramModel;

import java.io.IOException;
import java.nio.file.*;
import java.util.stream.Stream;

import org.preprocessing.sentenceSplitting.Tokenizer;

public class UpdateNgramModel {
    public static void main(String[] args) {

        NGramModel model = new NGramModel();
        model.loadModelFromCSV("final_dict");

        Tokenizer tokenizer = new Tokenizer();
        tokenizer.removeSpaces(true);

        Path folderPath = Paths.get("h:/laptop/wikipediadumps"); // Replace with your folder path

        // Since count needs to be modified inside a lambda, we can use an array of size 1
        final int[] count = {0};  // Use an array to keep count as a mutable variable

        try (Stream<Path> paths = Files.list(folderPath).parallel()) { // Use parallel stream
            paths.filter(Files::isRegularFile)  // Only files, not directories
                 .forEach(file -> {
                    try (Stream<String> lines = Files.lines(file)) { // Use Files.lines for efficiency
                        lines.parallel().forEach(line -> {
                            model.updateSentence(tokenizer.matchWordsAndSymbols(line.toLowerCase()));
                        });
                    } catch (IOException e) {
                        System.err.println("Error reading file " + file + ": " + e.getMessage());
                    }

                    // Count file processing, not line count
                    count[0]++;

                    if (count[0] % 1000 == 0) {
                        System.out.println("Processed: " + count[0] + " files");
                    }
                 });
        } catch (IOException e) {
            System.err.println("Error reading folder: " + e.getMessage());
        }

        model.saveModelToCSV("final_dict/updated");
    }
}
