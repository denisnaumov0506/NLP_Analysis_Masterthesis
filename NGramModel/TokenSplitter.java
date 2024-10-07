package org.NGramModel;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.preprocessing.dictionary.WordDictionary;
import org.preprocessing.sentenceSplitting.Tokenizer;

import edu.stanford.nlp.patterns.surface.Token;
import weka.classifiers.trees.ht.Split;

public class TokenSplitter {

    // Memoization map
    private static Map<String, List<SplitResult>> memo = new HashMap<>();

    // Main method for testing
    public static void main(String[] args) throws Exception {
        String dictionaryPath = "Z:/completed_dict_v1.json";
        String dictionaryAbbrevPath = "Z:/completed_abbrev_v1.json";

        // Load dictionaries
        WordDictionary.loadDictionary(dictionaryPath, 0, true, false);
        Map<String, Integer> dictionary = WordDictionary.getLoadedDictionary();

        WordDictionary.loadDictionary(dictionaryAbbrevPath, 0, true, false);
        Map<String, Integer> dictionaryAbbrev = WordDictionary.getLoadedDictionary();

        // Load the CSV file
        List<String> reviews = new ArrayList<>();

        reviews.add("no problems with me ilike it");

        List<String[]> splits = new ArrayList<>();

        // Create an instance of the tokenizer
        Tokenizer tokenizer = new Tokenizer();
        tokenizer.removeSpaces(true);

        for (String review : reviews) {
            // Tokenize the text
            List<String> tokens = tokenizer.matchWordsAndSymbols(review);

            LinkedHashMap<String, List<SplitResult>> test = checkSplits(tokens, dictionary);

            for (Map.Entry<String, List<SplitResult>> entry : test.entrySet()) {
                for (SplitResult result : entry.getValue()) {
                    List<String> abc = split(tokens, entry.getKey(), result);
    
                    System.out.println(Arrays.toString(abc.toArray(new String[0])));
                }
            }

            // for (String token : tokens) {
            //     // Find the best splits
            //     List<SplitResult> bestSplits = findBestSplits(token, dictionary);
            //     if (bestSplits != null && !bestSplits.isEmpty()) {
            //         if (bestSplits.get(0).totalFrequency > 100_000_000) {
            //             splits.add(new String[]{token, bestSplits.get(0).toString()});
            //         }
            //     }
            // }
        }

        // Write the splits to a file
        try (PrintWriter writer = new PrintWriter("compounds.txt", "UTF-8")) {
            for (String[] items : splits) {
                writer.println(items[0] + ": " + items[1]);
            }
        }
    }

    public static LinkedHashMap<String, List<SplitResult>> checkSplits(List<String> tokens, Map<String, Integer> dictionary) {
        LinkedHashMap<String, List<SplitResult>> splits = new LinkedHashMap<>();
    
        for (int i = 0; i < tokens.size(); i++) {
            String token = tokens.get(i);
            List<SplitResult> bestSplits = findBestSplits(tokens.get(i), dictionary);
            if (bestSplits != null && !bestSplits.isEmpty()) {
                String token_id = ""+i; // Assuming the token itself is the key

                if (splits.containsKey(token_id)) {
                    splits.computeIfPresent(token_id, (key, value) -> {
                        value.addAll(bestSplits); // Add all the elements from bestSplits
                        return value;
                    });
                } else {
                    splits.put(token_id, new ArrayList<>(bestSplits)); // Directly add bestSplits
                }
            }
        }
        return splits;
    }

    public static List<String> split(List<String> tokens, String key, SplitResult fixes) {

        List<String> newFixes = fixes.words;
        
        List<String> newTokens = new ArrayList<>();
        int id = Integer.parseInt(key.split(" ")[0]);

        List<String> preText = tokens.subList(0, id);
        List<String> sufText = tokens.subList(id + 1, tokens.size());

        newTokens.addAll(preText);
        newTokens.addAll(newFixes);
        newTokens.addAll(sufText);

        return newTokens;
    }
    

    // Function to find the best splits for a token
    public static List<SplitResult> findBestSplits(String token, Map<String, Integer> dictionary) {
        if (memo.containsKey(token)) {
            return memo.get(token);
        }

        // Try splitting into exactly two words first
        if (token.length() > 2) {
            List<SplitResult> bestTwoWordSplits = findSplitsWithNWords(token, dictionary, 2);
            if (!bestTwoWordSplits.isEmpty()) {
                memo.put(token, bestTwoWordSplits);
                return bestTwoWordSplits;
            }
        }


        // If no two-word splits, try splitting into three words
        if (token.length() > 3) {
            List<SplitResult> bestThreeWordSplits = findSplitsWithNWords(token, dictionary, 3);
            if (!bestThreeWordSplits.isEmpty()) {
                memo.put(token, bestThreeWordSplits);
                return bestThreeWordSplits;
            }
        }

        // Continue increasing the number of subwords if needed
        List<SplitResult> bestMultiWordSplits = new ArrayList<>();
        for (int n = 4; n <= token.length(); n++) {
            if (token.length() > n) {
                bestMultiWordSplits = findSplitsWithNWords(token, dictionary, n);
                if (!bestMultiWordSplits.isEmpty()) {
                    break;
                }
            }
        }

        memo.put(token, bestMultiWordSplits);
        return bestMultiWordSplits;
    }

    // Function to find splits with exactly n words
    private static List<SplitResult> findSplitsWithNWords(String token, Map<String, Integer> dictionary, int n) {
        if (n == 2) {
            // Try splitting the token into exactly two words
            List<SplitResult> results = new ArrayList<>();
            for (int i = 1; i < token.length(); i++) {
                String left = token.substring(0, i);
                String right = token.substring(i);

                long leftFreq = getFrequency(left, dictionary);
                long rightFreq = getFrequency(right, dictionary);

                if (leftFreq > 0 && rightFreq > 0) {
                    results.add(new SplitResult(Arrays.asList(left, right), leftFreq + rightFreq));
                }
            }
            return results;
        } else {
            return findSplitsRecursively(token, dictionary, n);
        }
    }

    // Recursive function to find splits with n words
    private static List<SplitResult> findSplitsRecursively(String token, Map<String, Integer> dictionary, int n) {
        List<SplitResult> results = new ArrayList<>();
        if (n == 1) {
            // Base case: if the whole token is a word
            long tokenFreq = getFrequency(token, dictionary);
            if (tokenFreq > 0) {
                results.add(new SplitResult(Collections.singletonList(token), tokenFreq));
            }
            return results;
        }

        // Try all possible splits for n words
        for (int i = 1; i < token.length(); i++) {
            String left = token.substring(0, i);
            String right = token.substring(i);

            long leftFreq = getFrequency(left, dictionary);
            if (leftFreq > 0) {
                List<SplitResult> rightSplits = findSplitsRecursively(right, dictionary, n - 1);

                for (SplitResult rightSplit : rightSplits) {
                    List<String> combinedWords = new ArrayList<>();
                    combinedWords.add(left);
                    combinedWords.addAll(rightSplit.words);

                    results.add(new SplitResult(combinedWords, leftFreq + rightSplit.totalFrequency));
                }
            }
        }
        return results;
    }

    // Function to get the frequency of a word from the dictionary
    private static long getFrequency(String word, Map<String, Integer> dictionary) {
        String lower = word.toLowerCase();
        if (dictionary.containsKey(lower)) {
            return (long) dictionary.get(lower);
        }
        return 0;
    }

    // public static List<String> splitWord(String word, Map<String, Integer> dict) {
        
    //     List<SplitResult> results = findBestSplits(word, dict);


    // }
}