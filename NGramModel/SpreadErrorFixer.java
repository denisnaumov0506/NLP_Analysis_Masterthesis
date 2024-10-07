package org.NGramModel;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.preprocessing.dictionary.WordDictionary;
import org.preprocessing.sentenceSplitting.Tokenizer;

public class SpreadErrorFixer {

    // Memoization map to avoid redundant checks
    private static Map<String, Boolean> memo = new HashMap<>();

    // Main method for testing
    public static void main(String[] args) throws Exception {
        String dictionaryPath = "Z:/completed_dict_v1.json";
        String dictionaryAbbrevPath = "Z:/completed_abbrev_v1.json";

        // Load dictionaries
        WordDictionary.loadDictionary(dictionaryPath, 0, true, false);
        Map<String, Integer> dictionary = WordDictionary.getLoadedDictionary();

        WordDictionary.loadDictionary(dictionaryAbbrevPath, 0, true, false);
        Map<String, Integer> dictionaryAbbrev = WordDictionary.getLoadedDictionary();

        // Load the CSV file or a list of sample reviews
        List<String> reviews = new ArrayList<>();
        reviews.add("ar e you insane");
        reviews.add("wh at i s happening");
        reviews.add("ca n you do this");
        reviews.add("My first order and my last v ery poor customer service");

        List<String[]> fixes = new ArrayList<>();

        // Create an instance of the tokenizer
        Tokenizer tokenizer = new Tokenizer();
        tokenizer.removeSpaces(true);

        for (String review : reviews) {
            // Tokenize the text
            List<String> tokens = tokenizer.matchWordsAndSymbols(review);

            // Check for spread errors by combining bigrams and verifying in the dictionary
            Map<String, List<String>> potentialFixes = checkBigrams(tokens, dictionary);
            if (!potentialFixes.isEmpty()) {
                for (Map.Entry<String, List<String>> fix : potentialFixes.entrySet()) {
                    fixes.add(new String[]{fix.getKey(), Arrays.toString(fix.getValue().toArray(new String[0]))});
                }
            }

            for (Map.Entry<String, List<String>> entries : potentialFixes.entrySet()) {
                List<String> newTokens = combine(tokens, entries.getKey());

                System.out.println(Arrays.toString(newTokens.toArray(new String[0])));
            }
        }

        // Write the potential fixes to a file
        try (PrintWriter writer = new PrintWriter("potential_bigrams_fixes.txt", "UTF-8")) {
            for (String[] items : fixes) {
                writer.println("Original: " + items[0] + " | Fix: " + items[1]);
            }
        }
    }

    public static List<String> combine(List<String> tokens, String key) {
        
        List<String> newTokens = new ArrayList<>();
        int id_pre = Integer.parseInt(key.split(" ")[0]);
        int id_suf = Integer.parseInt(key.split(" ")[1]);

        List<String> preText = tokens.subList(0, id_pre);
        List<String> fixText = new ArrayList<>(tokens.subList(id_pre, id_suf + 1));
        String combinedString = String.join("", fixText);
        fixText.clear();
        fixText.add(combinedString);
        List<String> sufText = tokens.subList(id_suf + 1, tokens.size());

        newTokens.addAll(preText);
        newTokens.addAll(fixText);
        newTokens.addAll(sufText);

        return newTokens;
    }

    public static List<String> combineWithExtraLetter(List<String> tokens, String key, String letter) {
        List<String> newTokens = new ArrayList<>();
        int id_pre = Integer.parseInt(key.split(" ")[0]);
        int id_suf = Integer.parseInt(key.split(" ")[1]);

        List<String> preText = tokens.subList(0, id_pre);
        List<String> fixText = new ArrayList<>(tokens.subList(id_pre, id_suf + 1));
        String combinedString = fixText.get(0) + letter + fixText.get(1);
        fixText.clear();
        fixText.add(combinedString);
        List<String> sufText = tokens.subList(id_suf + 1, tokens.size());

        newTokens.addAll(preText);
        newTokens.addAll(fixText);
        newTokens.addAll(sufText);

        return newTokens;
    }

    // Function to check all possible bigrams in tokens and find valid combinations in the dictionary
    public static LinkedHashMap<String, List<String>> checkBigrams(List<String> tokens, Map<String, Integer> dictionary) {
        // List<String> potentialFixes = new ArrayList<>();
        LinkedHashMap<String, List<String>> potentialFixesMap = new LinkedHashMap<String, List<String>>();
        
        // Iterate through tokens and combine bigrams (pairs of tokens)
        for (int i = 0; i < tokens.size() - 1; i++) {
            String combined = tokens.get(i) + tokens.get(i + 1);
            String combinedKey = i + " " + (i + 1);
            if (isValidWord(combined, dictionary)) {
                if (potentialFixesMap.containsKey(combinedKey)) {
                    potentialFixesMap.computeIfPresent(combinedKey, (key, value) -> {
                        value.add(combined);
                        return value;
                    });
                } else if (!potentialFixesMap.containsKey(combined)) {
                    List<String> arrayList = new ArrayList<>();
                    arrayList.add(combined);
                    potentialFixesMap.put(combinedKey, arrayList);
                }
            }
        }
        
        return potentialFixesMap;
    }

    // Function to check if a word is valid based on the dictionaries
    private static boolean isValidWord(String word, Map<String, Integer> dictionary) {
        if (memo.containsKey(word)) {
            return memo.get(word);
        }

        String lower = word.toLowerCase();
        boolean isValid = dictionary.containsKey(lower) && dictionary.get(lower) > 1_000_000;

        memo.put(word, isValid);
        return isValid;
    }
}
