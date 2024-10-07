package org.NGramModel;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.io.File;

import org.CustomExceptions.DictionaryNotFoundException;
import org.preprocessing.ErrorDetection.TypoDetector;
import org.preprocessing.dictionary.DeleteDictionary;
import org.preprocessing.dictionary.WordDictionary;
import org.preprocessing.sentenceSplitting.Tokenizer;
import org.utilities.NLPUtilities;

import com.fasterxml.jackson.core.exc.StreamReadException;
import com.fasterxml.jackson.databind.DatabindException;
import com.fasterxml.jackson.databind.ObjectMapper;

public class NGramModel {
    private static final int BUFFER_SIZE = 1024 * 1024; // 1 MB buffer size, adjust as necessary
    private HashMap<String, Integer> bigramCounts;
    private HashMap<String, Integer> unigramCounts;
    private HashMap<String, Integer> trigramCounts;
    private long totalWords;
    private long totalBigrams;
    private long totalTrigrams;
    private long unigramSize;
    private TopSuggestions suggestionScores;

    // // Cache to store already calculated n-gram probabilities
    // private Map<String, Double> probCache = new HashMap<>();
    // private Map<String, Double> editCache = new HashMap<>();
    // private Map<List<String>, Double> seqProbCache = new HashMap<>();
    // private Map<String, Boolean> pluralCache = new HashMap<>();
    // private Map<String, Boolean> typoCache = new HashMap<>();
    // private Map<String, Boolean> firstCache = new HashMap<>();

    public NGramModel() {
        this.bigramCounts = new HashMap<>();
        this.unigramCounts = new HashMap<>();
        this.trigramCounts = new HashMap<>();
        this.unigramSize = 0;
        this.suggestionScores = new TopSuggestions();
    }

    public void loadModelFromCSV(String filePath) {
        loadModelFromCSVOLD(filePath + "/uncased_combined_unigrams.csv", unigramCounts, "Uni-gram");
        loadModelFromCSVOLD(filePath + "/uncased_combined_bigrams.csv", bigramCounts, "Bi-gram");
        loadModelFromCSVOLD(filePath + "/uncased_combined_trigrams.csv", trigramCounts, "Tri-gram");
    }

    private void loadNGramWithMappedBuffer(String path, Map<String, Integer> ngramCounts, String ngramType) {
        System.out.println("Reading... "+ ngramType);
        try (RandomAccessFile file = new RandomAccessFile(path, "r");
             FileChannel channel = file.getChannel()) {

            long fileSize = channel.size();
            long position = 0;
            MappedByteBuffer buffer;
            boolean isFirstLine = true;  // To skip the header

            while (position < fileSize) {
                long size = Math.min(BUFFER_SIZE, fileSize - position); // Map up to BUFFER_SIZE
                buffer = channel.map(FileChannel.MapMode.READ_ONLY, position, size);
                position += size;

                while (buffer.hasRemaining()) {
                    // Read each line
                    StringBuilder line = new StringBuilder();
                    while (buffer.hasRemaining()) {
                        char ch = (char) buffer.get();
                        if (ch == '\n') break;  // End of line
                        line.append(ch);
                    }

                    // Process the line, skip headers
                    String lineStr = line.toString().trim();

                    // Skip first line if it's the header
                    if (isFirstLine) {
                        isFirstLine = false;
                        continue;
                    }

                    if (lineStr.isEmpty()) continue;

                    String[] parts = lineStr.split(",");
                    if (parts.length == 2) {
                        String ngram = parts[0];
                        int count = Integer.parseInt(parts[1]);  // Parse count as an integer
                        ngramCounts.put(ngram, count);  // Store in the model
                    }
                }
            }

            // Log the loading of the model
            System.out.println(ngramType + " model loaded from " + path);

            if (ngramType.equals("Uni-gram")) {
                // Calculate the totalWords
                totalWords = unigramCounts.values()
                    .stream()
                    .mapToLong(Integer::longValue)
                    .sum();
                this.unigramSize = unigramCounts.size();
                System.out.println("Calculated totalWords");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadModelFromCSVOLD(String path, Map<String, Integer> ngramCounts, String ngramType) {

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            boolean isFirstLine = true;  // Skip the header
            while ((line = br.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue;
                }
                String[] parts = line.split(",");
                if (parts.length == 2) {
                    String ngram = parts[0];
                    int count = Integer.parseInt(parts[1]);
                    ngramCounts.put(ngram, count);  // Add back to the model
                }
            }

            System.out.println("N-gram model loaded from " + path);
        } catch (IOException i) {
            i.printStackTrace();
        }

        totalWords = unigramCounts.values()
                        .stream()
                        .mapToLong(Integer::longValue)
                        .sum();
        this.unigramSize = unigramCounts.size();
    }

    // Add unigrams, bigrams, and trigrams to the model
    public void addSentence(List<String> sentence) {
        List<String> cleanSentence = new ArrayList<>();
        for (String word : sentence) {
            word = NLPUtilities.keepLatinTextOnly(word);

            if (!NLPUtilities.hasLatinAlphabet(word)) {
                continue;
            }

            word = NLPUtilities.cleanWord(word);

            cleanSentence.add(word);
        }

        for (int i = 0; i < cleanSentence.size() - 1; i++) {
            String word = cleanSentence.get(i);
            String nextWord = cleanSentence.get(i + 1);
            String thirdWord = null;
            if (cleanSentence.size() >= 3 && i < cleanSentence.size() - 2)
                thirdWord = cleanSentence.get(i + 2);

            // Increment unigram count
            unigramCounts.put(word, unigramCounts.getOrDefault(word, 0) + 1);

            // Create the bigram key
            String bigram = word + " " + nextWord;
            bigramCounts.put(bigram, bigramCounts.getOrDefault(bigram, 0) + 1);

            // Create the trigram key
            if (thirdWord != null) {
                String trigram = word + " " + nextWord + " " + thirdWord;
                trigramCounts.put(trigram, trigramCounts.getOrDefault(trigram, 0) + 1);
            }
        }
    }

    // Update unigrams, bigrams, and trigrams of the loaded model
    public void updateSentence(List<String> sentence) {
        List<String> cleanSentence = new ArrayList<>();
        for (String word : sentence) {
            word = NLPUtilities.keepLatinTextOnly(word);

            if (!NLPUtilities.hasLatinAlphabet(word)) {
                continue;
            }

            word = NLPUtilities.cleanWord(word);

            cleanSentence.add(word);
        }

        for (int i = 0; i < cleanSentence.size() - 1; i++) {
            String word = cleanSentence.get(i);
            String nextWord = cleanSentence.get(i + 1);
            String thirdWord = null;
            if (cleanSentence.size() >= 3 && i < cleanSentence.size() - 2)
                thirdWord = cleanSentence.get(i + 2);

            // Increment unigram count
            if (unigramCounts.containsKey(word)) {
                unigramCounts.put(word, unigramCounts.getOrDefault(word, 0) + 1);
            }

            // Create the bigram key
            String bigram = word + " " + nextWord;
            if (bigramCounts.containsKey(bigram)) {
                bigramCounts.put(bigram, bigramCounts.getOrDefault(bigram, 0) + 1);
            }

            // Create the trigram key
            if (thirdWord != null) {
                String trigram = word + " " + nextWord + " " + thirdWord;
                if (trigramCounts.containsKey(trigram)) {
                    trigramCounts.put(trigram, trigramCounts.getOrDefault(trigram, 0) + 1);
                }
            }
        }
    }

    public void saveModelToCSV(String filePath) {
        try (FileWriter csvWriter = new FileWriter(filePath + "/bigram_model.csv")) {
            // Write headers
            csvWriter.append("Bigram,Count\n");

            // Write bigram counts
            for (Map.Entry<String, Integer> entry : bigramCounts.entrySet()) {
                csvWriter.append(entry.getKey()).append(",").append(entry.getValue().toString()).append("\n");
            }

            System.out.println("Bi-gram model saved to " + filePath + "/bigram_model.csv");
        } catch (IOException i) {
            i.printStackTrace();
        }

        try (FileWriter csvWriter = new FileWriter(filePath + "/trigram_model.csv")) {
            // Write headers
            csvWriter.append("Trigram,Count\n");

            // Write trigram counts
            for (Map.Entry<String, Integer> entry : trigramCounts.entrySet()) {
                csvWriter.append(entry.getKey()).append(",").append(entry.getValue().toString()).append("\n");
            }

            System.out.println("Tri-gram model saved to " + filePath + "/trigram_model.csv");
        } catch (IOException i) {
            i.printStackTrace();
        }

        try (FileWriter csvWriter = new FileWriter(filePath + "/unigram_model.csv")) {
            // Write headers
            csvWriter.append("Unigram,Count\n");

            // Write bigram counts
            for (Map.Entry<String, Integer> entry : unigramCounts.entrySet()) {
                csvWriter.append(entry.getKey()).append(",").append(entry.getValue().toString()).append("\n");
            }

            System.out.println("Uni-gram model saved to " + filePath + "/unigram_model.csv");
        } catch (IOException i) {
            i.printStackTrace();
        }
    }

    public Map<String, Integer> getUnigram() {
        return unigramCounts;
    }

    // public double getSequenceProbabilityDynamic(List<String> tokens, boolean verbose) {
    //     double probability = 0.0;
    //     double trigramWeight = 0.7;
    //     double bigramWeight = 0.2999;
    //     double unigramWeight = 0.0001;
    //     double epsilon = 1e-10; // Small value to avoid log(0)

    //     for (int i = 0; i < tokens.size(); i++) {
    //         String unigram = tokens.get(i);
    //         long unigramCount = unigramCounts.getOrDefault(unigram, 0);
    //         double unigramProb = ((double) unigramCount + epsilon) / (totalWords + unigramSize);

    //         if (i > 0) {
    //             String bigram = tokens.get(i - 1) + " " + unigram;
    //             long bigramCount = bigramCounts.getOrDefault(bigram, 0);
    //             double bigramProb = (bigramCount > 0) ? (double) bigramCount / unigramCounts.getOrDefault(tokens.get(i - 1), 1) : unigramProb;

    //             if (i > 1) {
    //                 String trigram = tokens.get(i - 2) + " " + tokens.get(i - 1) + " " + unigram;
    //                 long trigramCount = trigramCounts.getOrDefault(trigram, 0);
    //                 double trigramProb = (trigramCount > 0) ? (double) trigramCount / bigramCounts.getOrDefault(tokens.get(i - 2) + " " + tokens.get(i - 1), 1) : bigramProb;
    //                 probability += trigramWeight * Math.log(trigramProb + epsilon);
    //             } else {
    //                 probability += bigramWeight * Math.log(bigramProb + epsilon);
    //             }
    //         } else {
    //             probability += unigramWeight * Math.log(unigramProb + epsilon);
    //         }
    //     }

    //     return Math.exp(probability/tokens.size());
    // }

    public double getSequenceProbabilityDynamic(List<String> tokens, boolean verbose) {

        // verbose = true;

        double prob = 1.0;
        double trigramAlpha = 8; // 1 (8 - 0.000005)/(8 - 0.000005)
        double bigramAlpha = 0.05; // (0.05 - 0.000005)/(8 - 0.000005)
        double unigramAlpha = 0.000005; // (0.000005 - 0.)

        for (int i = 0; i < tokens.size(); i++) {

            // if ((tokens.get(2).equals("like")) && tokens.get(2+1).equals("as") && tokens.get(4).equals("am")) {
            //     System.out.println();
            // }

            // if ((tokens.get(2).equals("like")) && tokens.get(2+1).equals("a") && tokens.get(4).equals("s")) {
            //     System.out.println();
            // }

            String unigram = tokens.get(i);
            long uniCount = unigramCounts.getOrDefault(unigram, 1);

            String bigram = null;
            String trigram = null;
            long countBi = 0;
            long countTri = 0;

            if (i>1) {
                bigram = tokens.get(i - 1) + " " + unigram;
                countBi = bigramCounts.getOrDefault(bigram, 0);
            }
            if (i>2) {
                trigram = tokens.get(i - 2) + " " + bigram;
                countTri = trigramCounts.getOrDefault(trigram, 0);
            }

            if (i == 0) {
                // Unigram probability for the first word
                long count = uniCount;

                // double unigramImpact = (double) count / (double) totalWords;

                // Unigrams should receive the hardest penalty as they are sometimes wrong,
                // to mitigate the impact of very large frequency words like stop words.
                double uniProb = 0.0;
                // if (probCache.containsKey(unigram)) {
                //     uniProb = probCache.get(unigram);
                // } else {
                //     probCache.put(unigram, uniProb);
                // }
                uniProb = ((double) count / (double) (totalWords));
                prob *= unigramAlpha * uniProb;

                if (verbose) {
                    System.out.println(tokens.get(i) + "(" + prob + ")");
                }
            } else if (i == 1) {
                if (countBi > 0) {
                    // Use bigram if available
                    double biProb = 0.0;
                    long countUni = unigramCounts.getOrDefault(tokens.get(i - 1), 1);
                    biProb = ((double) countBi / (double) (countUni + this.unigramSize));
                    // if (probCache.containsKey(bigram)) {
                    //     biProb = probCache.get(bigram);
                    // } else {

                    //     probCache.put(bigram, biProb);
                    // }

                    prob *= bigramAlpha * biProb;
                    if (verbose) {
                        System.out.println(bigram + "(" + prob + ")");
                    }
                } else {
                    // Fallback to unigram if bigram is not available

                    double uniProb = 0.0;
                    long count = uniCount;
                    uniProb = ((double) count / (double) (totalWords));
                    // if (probCache.containsKey(unigram)) {
                    //     uniProb = probCache.get(unigram);
                    // } else {

                    //     probCache.put(unigram, uniProb);
                    // }

                    // Unigrams should receive the hardest penalty such that bigrams and trigrams should be preferred,
                    // to mitigate the impact of very large frequency words like stop words.
                    prob *= unigramAlpha * uniProb;

                    if (verbose) {
                        System.out.println(unigram + "(" + prob + ")");
                    }
                }
            } else {
                // Attempt trigram first
                if (countTri > 0) {
                    double triProb = 0.0;
                    String bigramPrev = tokens.get(i - 2) + " " + tokens.get(i - 1);
                    long countBiPrev = bigramCounts.getOrDefault(bigramPrev, 1);
                    triProb = ((double) countTri / (double) (countBiPrev + this.unigramSize));
                    // if (probCache.containsKey(trigram)) {
                    //     triProb = probCache.get(trigram);
                    // } else {

                    //     probCache.put(trigram, triProb);
                    // }
                    // Use trigram if available

                    // Trigrams need to receive a bonus since their overall frequency has a tendency to be lower then the bigram counterparts

                    prob *= trigramAlpha * triProb;

                    if (verbose) {
                        System.out.println(trigram + "(" + prob + ")");
                    }
                } else {
                    // Fallback to bigram if trigram is not available

                    if (countBi > 0) {
                        // Use bigram if available
                        double biProb = 0.0;
                        long countUni = unigramCounts.getOrDefault(tokens.get(i - 1), 1);
                        biProb = ((double) countBi / (double) (countUni + this.unigramSize));
                        // if (probCache.containsKey(bigram)) {
                        //     biProb = probCache.get(bigram);
                        // } else {

                        //     probCache.put(bigram, biProb);
                        // }

                        prob *= bigramAlpha * biProb;

                        if (verbose) {
                            System.out.println(bigram + "(" + prob + ")");
                        }
                    } else {
                        double uniProb = 0.0;
                        uniProb = ((double) uniCount / (double) (totalWords));
                        // if (probCache.containsKey(unigram)) {
                        //     uniProb = probCache.get(unigram);
                        // } else {

                        //     probCache.put(unigram, uniProb);
                        // }
                        prob *= unigramAlpha * uniProb;
                        // Fallback to unigram if bigram is not available

                        if (verbose) {
                            System.out.println(unigram + "(" + prob + ")");
                        }
                    }
                }
            }
        }
        return prob; // this gives us the geometric mean of the probability
    }

    // Evaluation function
    public void evaluate(List<List<String>> tokensCorrected, List<List<String>> tokensTrue) {
        int correctCount = 0;
        int tokenCount = 0;



        for (int j = 0; j < tokensTrue.size(); j++) {

            tokenCount++;

            if (tokensCorrected.get(j).size() != tokensTrue.get(j).size()) {
                continue;
            } else {
                boolean same = tokensCorrected.get(j).equals(tokensTrue.get(j));
                if (same) {
                    correctCount++;
                }
            }

            for (int i = 0; i < tokensTrue.get(j).size(); i++) {
                // tokenCount++;
                if (tokensCorrected.get(j).get(i).equals(tokensTrue.get(j).get(i))) {
                    // correctCount++;
                } else {
                    System.out.println("Mismatch: " + tokensCorrected.get(j));
                    System.out.println("Mismatch: " + tokensCorrected.get(j).get(i) + " != " + tokensTrue.get(j).get(i));
                }
            }
        }


        double accuracy = (double) correctCount / tokenCount;
        System.out.println("Accuracy: " + accuracy * 100 + "%");
    }

    public void updateSpread(List<String> tokens, int idx, TopSuggestions suggestionScores) {
        double editDistance = -1;
        double editDistanceValue = -1;

        boolean isMisplacement = true;
        boolean isOmit = true;
        boolean isAdded = true;
        boolean isSwap = true;
        boolean isSimilar = true;
        boolean isPlural = true;

        boolean isFirst = true;

        boolean isApostrophe = true;

        boolean isTypo = isMisplacement || isOmit || isAdded || isSwap || isSimilar;

        double epsilon = 0.8;

        double combinedScore = 0.0;

        if (idx < tokens.size() - 1) {
            String spreadWord = null;
            spreadWord = tokens.get(idx) + tokens.get(idx+1);
            List<String> newTokensSpread = SpreadErrorFixer.combine(tokens, idx+" "+(idx+1));
            double probSpread = 0.0;
            if (unigramCounts.containsKey(spreadWord)) {
                probSpread = getSequenceProbabilityDynamic(newTokensSpread, false);
            }

            editDistance = 0;
            editDistanceValue = 1 - (editDistance / 1);

            isMisplacement = true;
            isOmit = true;
            isAdded = true;
            isSwap = true;
            isSimilar = true;

            isFirst = true;

            isTypo = isMisplacement || isOmit || isAdded || isSwap || isSimilar;

            epsilon = 0.8;

            combinedScore = probSpread * (editDistance + epsilon) - (!isTypo ? 0.00 : 0.00);

            if (probSpread > 0.0) {
                updateScores(suggestionScores, newTokensSpread, probSpread, "spread");
            }
        }
    }

    public void updateSpreadX(List<String> tokens, int idx, TopSuggestions suggestionScores) {
        List<String> letters = new ArrayList<>(Arrays.asList("x", "c", "v", "b", "n", "m"));

        if (idx < tokens.size() - 1) {
            for (int e = 0; e < letters.size(); e++) {
                String spreadWordWithExtraLetter = null;
                spreadWordWithExtraLetter = tokens.get(idx) + letters.get(e) + tokens.get(idx+1);
                List<String> newTokensSpreadWithExtraLetter = SpreadErrorFixer.combineWithExtraLetter(tokens, idx+" "+(idx+1), letters.get(e));
                double probSpreadWithExtraLetter = 0.0;
                if (unigramCounts.containsKey(spreadWordWithExtraLetter)) {
                    probSpreadWithExtraLetter = getSequenceProbabilityDynamic(newTokensSpreadWithExtraLetter, false);
                }

                double editDistance = 0.0;
                double editDistanceValue = 1 - (editDistance / 1);

                boolean isMisplacement = true;
                boolean isOmit = true;
                boolean isAdded = true;
                boolean isSwap = true;
                boolean isSimilar = true;

                boolean isFirst = true;

                boolean isTypo = isMisplacement || isOmit || isAdded || isSwap || isSimilar;

                double epsilon = 0.8;

                double combinedScore = probSpreadWithExtraLetter * (editDistance + epsilon) - (!isTypo ? 0.00 : 0.00);

                if (probSpreadWithExtraLetter > 0.0) {
                    updateScores(suggestionScores, newTokensSpreadWithExtraLetter, probSpreadWithExtraLetter, "spread");
                }
            }
        }
    }

    public void updateCompound(List<String> tokens, int idx, TopSuggestions suggestionScores) {
        List<SplitResult> compoundWords = TokenSplitter.findBestSplits(tokens.get(idx), unigramCounts);

        for (int j = 0; j < compoundWords.size(); j++) {
            List<String> newTokensCompound = TokenSplitter.split(tokens, ""+idx, compoundWords.get(j));

            String compounds = String.join(" ", compoundWords.get(j).getWords());

            double probCompound = getSequenceProbabilityDynamic(newTokensCompound, false);

            double editDistance = 0.0;
            double editDistanceValue = 1 - (editDistance / 1);

            boolean isMisplacement = true;
            boolean isOmit = true;
            boolean isAdded = true;
            boolean isSwap = true;
            boolean isSimilar = true;

            boolean isFirst = true;

            boolean isTypo = isMisplacement || isOmit || isAdded || isSwap || isSimilar;

            double epsilon = 0.8;

            double combinedScore = probCompound * (editDistance + epsilon) - (!isTypo ? 0.00 : 0.00);

            if (probCompound > 0.0) {
                updateScores(suggestionScores, newTokensCompound, probCompound, "compound");
            }
        }
    }

    public void updateSuggestion(List<String> tokens, int idx, TopSuggestions suggestionScores, Map<String, List<String>> deleteDict) {
        Set<String> suggestions = new HashSet<>();

        String error = tokens.get(idx);

        // if (error.equals("supporting")) {
        //     System.out.println("supporting");
        // }

        if (error.equals("cant")) {
            suggestions.add("can't");
        } else if (error.equals("couldnt")) {
            suggestions.add("couldn't");
        } else if (error.equals("shouldnt")) {
            suggestions.add("shouldn't");
        } else if (error.equals("wouldnt")) {
            suggestions.add("wouldn't");
        } else if (error.equals("isnt")) {
            suggestions.add("isn't");
        } else if (error.equals("arent")) {
            suggestions.add("aren't");
        } else if (error.equals("werent")) {
            suggestions.add("weren't");
        } else if (error.equals("wasnt")) {
            suggestions.add("wasn't");
        } else if (error.equals("havent")) {
            suggestions.add("haven't");
        } else if (error.equals("hasnt")) {
            suggestions.add("hasn't");
        } else if (error.equals("dont")) {
            suggestions.add("don't");
        } else if (error.equals("doesnt")) {
            suggestions.add("doesn't");
        } else if (error.equals("wont")) {
            suggestions.add("won't");
        } else if (error.equals("aint")) {
            suggestions.add("ain't");
        }
        else {
            // Now we obtain the suggestions
            Set<String> deleteEdits = DeleteDictionary.generateOneDeleteEdits(Set.of(error), false);

            updateSuggestions(deleteEdits, deleteDict, suggestions, error);

            deleteEdits.clear();
        }

        // System.out.println("Suggestion_sizes: " + suggestions.size());
        // int count = 0;

        for (String suggestion : suggestions) {

            // System.out.println(++count+" "+suggestions.size());
            // System.out.println(tokens);
            // System.out.println(suggestion);

            // if (suggestion.equals("the")) {
            //     System.out.println("the");
            // }

            String prev = ""+tokens.get(idx);
            tokens.set(idx, suggestion);

            double editDistance = 0.0;
            editDistance = WordDictionary.editDistance(error, suggestion);
            // if (editCache.containsKey(error+" "+suggestion+"_"+suggestion+" "+error)) {
            //     editDistance = editCache.get(error+" "+suggestion+"_"+suggestion+" "+error);
            // } else {
            //     editCache.put(error+" "+suggestion+"_"+suggestion+" "+error, editDistance);
            // }

            double editDistanceValue = 1 - (editDistance / 1);

            boolean isPlural = false;
            isPlural = TypoDetector.isPlural(error, suggestion);
            // if (pluralCache.containsKey(error+" "+suggestion+"_"+suggestion+" "+error)) {
            //     isPlural = pluralCache.get(error+" "+suggestion+"_"+suggestion+" "+error);
            // } else {
            //     pluralCache.put(error+" "+suggestion+"_"+suggestion+" "+error, isPlural);
            // }

            boolean isTypo = false;
            boolean isMisplacement = TypoDetector.isTypoDueToMisplacement(suggestion, error, 1.5);
            boolean isOmit = TypoDetector.omitMistake(error, suggestion) || TypoDetector.omitMistake(suggestion, error);
            boolean isAdded = TypoDetector.addedMistake(error, suggestion);
            boolean isSwap = TypoDetector.swapMistake(error, suggestion) || TypoDetector.swapMistake(suggestion, error);
            boolean isSimilar = TypoDetector.similarSpelling(error, suggestion) || TypoDetector.similarSpelling(suggestion, error);

            isTypo = isMisplacement || isOmit || isAdded || isSwap || isSimilar;
            // if (typoCache.containsKey(error+" "+suggestion+"_"+suggestion+" "+error)) {
            //     isTypo = typoCache.get(error+" "+suggestion+"_"+suggestion+" "+error);
            // } else {
            //     typoCache.put(error+" "+suggestion+"_"+suggestion+" "+error, isTypo);
            // }

            boolean isFirst = false;
            isFirst = TypoDetector.startsWithSameletter(error, suggestion);
            // if (firstCache.containsKey(error+" "+suggestion+"_"+suggestion+" "+error)) {
            //     isFirst = firstCache.get(error+" "+suggestion+"_"+suggestion+" "+error);
            // } else {
            //     firstCache.put(error+" "+suggestion+"_"+suggestion+" "+error, isFirst);
            // }

            double probSuggestion = 0.0;

            if (isTypo && editDistance <= 1) {
                // if (seqProbCache.containsKey(tokens)) {
                //     probSuggestion = seqProbCache.get(tokens);
                // } else {
                //     probSuggestion = getSequenceProbabilityDynamic(tokens, false);
                //     seqProbCache.put(tokens, probSuggestion);
                // }

                probSuggestion = getSequenceProbabilityDynamic(tokens, false);
            }

            // Put the old token back
            tokens.set(idx, prev);

            double combinedScore = getCombinedScore(probSuggestion, editDistance, isTypo, isFirst);

            if (probSuggestion > 0.0) {
                updateScores(suggestionScores, suggestion, probSuggestion, "suggestion");
            }

        }

        suggestions.clear();
    }

    private double getCombinedScore(double probSuggestion, double editDistance, boolean isTypo, boolean isFirst) {
        double epsilon = 0.8;
        return (probSuggestion * (editDistance + epsilon) - (!isTypo ? 0.10 : 0.00)) * (isFirst ? 1.5 : 1.0);
    }

    private void updateScores(TopSuggestions suggestionScores, Object suggestion, double prob, String type) {
        suggestionScores.addSuggestion(new Suggestion(suggestion, prob, type));
    }

    // private void sortCollection(List<List<Object>> suggestionScores) {
    //     // Depending on which token sequence is the largest we return that
    //     // Sort based on edit distance first, then score
    //     Collections.sort(suggestionScores, new Comparator<List<Object>>() {
    //         @Override
    //         public int compare(List<Object> s1, List<Object> s2) {
    //             return Double.compare((double) s2.get(1), (double) s1.get(1));
    //         }
    //     });
    // }

    private void normalizeCollection(TopSuggestions suggestionScores, double totalProbability) {
        // Normalize the probabilities
        for (Suggestion suggestion : suggestionScores.getTopSuggestions()) {
            double originalProbability = (double) suggestion.getScore();
            double normalizedProbability = originalProbability / totalProbability;
            suggestion.updateScore(normalizedProbability);  // Replace the old probability with the normalized one
        }
    }

    private double getTotalProb(TopSuggestions suggestionScores) {
        // First, calculate the sum of all probabilities for normalization
        double totalProbability = 0.0;
        for (Suggestion suggestion : suggestionScores.getTopSuggestions()) {
            totalProbability += (double) suggestion.getScore();  // Assuming the probability is stored at index 4
        }

        return totalProbability;
    }

    private void updateSuggestions(Set<String> deleteEdits, Map<String, List<String>> deleteDict, Set<String> suggestions, String error) {
        for (String delete : deleteEdits) {
            if (deleteDict.containsKey(delete)) {
                suggestions.addAll(deleteDict.get(delete));
            }
        }

        if (deleteDict.containsKey(error)) {
            suggestions.addAll(deleteDict.get(error));
        }

        suggestions.add(error);
    }

    /**
     * This method corrects spelling errors.
     * @param tokens tokenized text
     * @param deleteDict dictionary containing suggestions
     * @param verbose Shows probabilities for each replacement
     * @param threshold The threshold for the model to do replacements. The higher, the less risky, it will be
     * @param freqThreshold The treshold of the frequency of token that determines whether the model considers it an error or not
     * @param exceptionList A list of tokens to consider in spell checking
     * @return
     */
    public List<String> spell(List<String> tokens, Map<String, List<String>> deleteDict, boolean verbose, double threshold, long freqThreshold, List<String> exceptionList) {

        // Exception list contains all tokens that should be evaluated for contextual appropriateness!
        // Preprocess all tokens
        for (int i = 0; i < tokens.size(); i++) {
            if (!NLPUtilities.hasLatinAlphabet(tokens.get(i))) {
                continue;
            } else {
                tokens.set(i, tokens.get(i).strip());
            }
        }

        // First obtain the compound error, spread error and all suggestions for a token
        for (int i = 0; i < tokens.size(); i++) {

            // if (tokens.get(i).equals("v")) {
            //     System.out.println("v");
            // }

            boolean skip = false;

            // Preprocess
            if (!NLPUtilities.hasLatinAlphabet(tokens.get(i))) {
                continue;
            }

            if (unigramCounts.containsKey(tokens.get(i))) {
                if (unigramCounts.get(tokens.get(i)) > freqThreshold && !exceptionList.contains(tokens.get(i))) {

                    skip = true; // Skip suggestion for higher frequency tokens, but only if they are not inside the exception list
                }
            } else {
                skip = true; // Skip if it is not inside unigramCounts
            }

            // First we need a empty the scores list
            suggestionScores.clear();

            String error = tokens.get(i);

            updateSpread(tokens, i, suggestionScores);
            updateSpreadX(tokens, i, suggestionScores);
            updateCompound(tokens, i, suggestionScores);

            if (!skip) { // Only attempt to fix it if the skip flag is set
                updateSuggestion(tokens, i, suggestionScores, deleteDict);
            } else {
                double prob = getSequenceProbabilityDynamic(tokens, false);

                if (prob > 0.0) {
                    updateScores(suggestionScores, error, prob, "suggestion");
                }
            }

            // sortCollection(suggestionScores);

            // suggestionScores = suggestionScores.subList(0, Math.min(suggestionScores.size(), 10)); // Reduce the size of the candidates.

            double totalProbability = getTotalProb(suggestionScores);

            normalizeCollection(suggestionScores, totalProbability);

            Suggestion[] suggs = suggestionScores.getTopSuggestions(); // these are sorted!

            if (verbose) {
                System.out.println("Correction of: " + tokens.get(i));

                // Print or return the top 10 suggestions

                for (Suggestion suggestion : suggs) {
                    String type = (String) suggestion.getType();
                    String topSuggestion = null;
                    if (type.equals("spread") || type.equals("compound") || type.equals("spreadWithExtraLetter")) {
                        topSuggestion = Arrays.toString(((List<String>) suggestion.getSuggestion()).toArray(new String[0]));
                    } else {
                        topSuggestion = (String) suggestion.getSuggestion();
                    }
                    double topScore = (double) suggestion.getScore();
                    System.out.println("Suggestion: " + topSuggestion + ", Probability: " + topScore);
                }
            }

            // Replacement logic
            if (suggestionScores.getTopSuggestions().length > 0) {
                String type = (String) suggestionScores.getTopSuggestions()[0].getType();
                double prob = (double) suggestionScores.getTopSuggestions()[0].getScore();

                if (prob > threshold) {
                    if (type.equals("spread") || type.equals("spreadWithExtraLetter")) {
                        List<String> newTokenList = (List<String>) suggestionScores.getTopSuggestions()[0].getSuggestion();

                        tokens = newTokenList;

                        // check if we can still improve it by combining it with the next token if possible
                        // simply changing the i again
                        i = i - 1; // go back again!
                    } else if (type.equals("compound")) {
                        List<String> newTokenList = (List<String>) suggestionScores.getTopSuggestions()[0].getSuggestion();

                        tokens = newTokenList;
                    } else {
                        // Update the tokens list with the new top suggestions
                        String top = suggestionScores.getTopSuggestions().length > 0 ? (String) suggestionScores.getTopSuggestions()[0].getSuggestion() : error;

                        tokens.set(i, top); // this updates it!

                        if (!error.equals(top)) { // change has been applied, please backtrack to check again
                            // Did not end up using it, since this can cause infinite loops and potential overcorrection issues
                            
                            // // go back two tokens to evaluate the previous tokens again
                            // if (i >= 2) {
                            //     i -= 2;
                            // } else if (i >= 1) {
                            //     i -= 1;
                            // }
                        }
                    }
                } else {
                    // We simply do not change anything
                }
            } else {
                // We simply do not change anything
            }
        }
        return tokens;
    }

    public static void main(String[] args) throws DictionaryNotFoundException, StreamReadException, DatabindException, IOException {
        NGramModel nGramModel = new NGramModel();

        // Create an ObjectMapper instance
        ObjectMapper objectMapper = new ObjectMapper();

        // Read the JSON file and convert it back to a Map<String, List<String>>
        Map<String, List<String>> dict_deletes = objectMapper.readValue(new File("final_dict/delete_edits.json"), Map.class);

        System.out.println("Delete dict loaded");

        // nGramModel.loadModelFromCSVMemoryMapped("final_dict");
        nGramModel.loadModelFromCSV("final_dict");

        Tokenizer tokenizer = new Tokenizer();

        tokenizer.removeSpaces(false);

        List<String> sents = new ArrayList<>();

        // doordash
        sents.add("my first order and my last v ery poor customer service");
        sents.add("it sounds like a s am to me.");
        sents.add("i hate you door dash");
        sents.add("no problems with me ilike it");
        sents.add("had mikey help me out.");
        sents.add("best biryani, filter coffee and irani chai");
        sents.add("its handy but the price increses are insane");
        sents.add("saved my butt durring covid");
        sents.add("i have never had any problems until yesterday when the driver clearly stold my daughter's food, and when she called them they told her there was nothing they could do.");
        sents.add("it's ohkay");
        sents.add("still waiting on my refunded 10$ been over 7 days");
        sents.add("duplin county needs to build something besides walmart dollar generals and car washes !");
        sents.add("we need food delivery services, bowling, malls, skating rinks, movie theateres, barnes and noble book stores, krispy creme donut shops");
        sents.add("I waited 30min and got no response befofe i eventually gave up and ended the chat.");
        sents.add("This is not how you should be treating uour customers.");
        sents.add("first, they double charged me for everything and refused to reverse the extra charge untill i explained to them that this is canada and our rights are better than the states.");
        sents.add("greeat app.");
        sents.add("and the only thing i suggest is that the restaurant's pay more close attention to the order & make sure napkins and condiments are with the order");
        sents.add("i've had a few problem's here and there, but customer service always fixed the problem.");
        sents.add("almost always great service but i've had a few instances where dashers were really late and one time i paid for my order but it was not received by the restaurant for some reason, doordash customer services was great and offerd me a refund.");

        // twitch
        sents.add("its one of the best straming sites but 2 things are less good 1.");
        sents.add("is therea reason why i can't find my friend's account on here?");
        sents.add("i love everything about this app exept for the adds.");
        sents.add("definitely recomend!!");
        sents.add("love everthing about the app");
        sents.add("thank you for make the app.");
        sents.add("Devs keep taking away useful features and changing things like pip mode and audio only/background mode.");
        sents.add("what are you planning to take away next, the abilty to select a stream to watch?");
        sents.add("sometimes i want to chat and it says connectimg and i cant type");
        sents.add("the mobile website has wayy better quality");
        sents.add("so can you not chage your username on here or what?");
        sents.add("i wouldn't change anything about twitch, the app is just to good.");
        sents.add("i love the app but most of the time when i whatch a stream it pauses out and stops the stream when i whatch something, even if connected to a signal and it's only happening fro mobile...");
        sents.add("if this is just a big with the app, pls fix it");
        sents.add("it makes stuff super easy for many diffrent reasons a recomend it 100%");
        sents.add("one of the best streaming platfroms in my life");
        sents.add("the app is fine if the website wasnt better.");
        sents.add("the websits has completely different added features that the app just doesnt have.");
        sents.add("this app is awseome but how i can join with hem and when i talk on chats hey banned me");
        sents.add("tried at least 100 times on 3 different coccasions to link cod app with twich app and i keep getting errors.");

        // whatsapp
        sents.add("is very most essly use for photos, videos, calling, chating and more every things else lovely, amazing");
        sents.add("connects me with my frieds");
        sents.add("best app for chating with lover");
        sents.add("amazing plateform for +tive users");
        sents.add("best app for messaging and also usefull for the business");
        sents.add("very good and relable");
        sents.add("you delete the reviw");
        sents.add("dear watsapp team and management");
        sents.add("please solvee this problem");
        sents.add("great but neend some more exciting features");
        sents.add("it is very good quality appp");
        sents.add("whit out reason my acount is banned or going to review form");
        sents.add("it is good for andorid phones or tab and also in i phones but not in amazon tab at all");
        sents.add("iwant to instal two whats app in two sim");
        sents.add("its amazing it is very usful its very funny");
        sents.add("can we get atleast our different versions of the prev whatsapp so we csn have options to choose the 1 we enjoyed before updates?");
        sents.add("video call service wrost");
        sents.add("the whatsapp is amazinng platform");
        sents.add("superb app for comunication");
        sents.add("where is my acount?");

        // pokemongo
        sents.add("changing the avatars oprions for some dei stuff could be accepted, but the selling to option for the old avatars back at a premium is just the worst");
        sents.add("first of all the game is super buggy has been since release and no patches have come out yet you \\\"fix\\\" things people loved about the game.");
        sents.add("worest customer service,");
        sents.add("it said active and to login and ur quest will be there but nothing ,");
        sents.add("update fixed the game, i would love for you to add a recieve all and send all button for gifts, sending individually is pure pain.");
        sents.add("big pokemon are too far away");
        sents.add("you never get what you need to catch and it's almost impossible to catch biggerr scale pokemon");
        sents.add("really good i will definitely recomend to others");
        sents.add("pokemon go is fun");
        sents.add("we don't care if it's hard to find freinds");
        sents.add("you just need someone hwo play it");
        sents.add("dose the updates take alot of storage i can't play it enamored unless i delete everything on my phone");
        sents.add("veryy 'pay-to-win', so it doesn't really hold your interest for long.");
        sents.add("its a realy good game but sometimes you get board and you cant get pokemons around the city untill your at level 20");
        sents.add("bestq games in 2016");
        sents.add("pokman is a fun gamd");
        sents.add("its relly bad its soo lag at frst its good ef you get to level 20 on master leg the lag begens");
        sents.add("greate family fun");
        sents.add("i just don't like that there were no opt for \\\"claim all\\\" and \\\"send all\\\" in receving/sending gifts!");
        sents.add("fire brings back so many menories");

        // outlook
        sents.add("with this let me experience more to give 4 or 5 star's");
        sents.add("excellent srevice");
        sents.add("outlook allways work fine.");
        sents.add("upgraded my phine to a samsung galaxy s22 and the outlook new mail sound is no longer the microsoft new mail sound and i can only change it to what is available on the phone.");
        sents.add("great app, keeps me updated without having to logg into my pc");
        sents.add("real sasy to use");
        sents.add("simole, fast and efficient");
        sents.add("enjoye empty inbox feature is not compatible with us.");
        sents.add("please trll how to keep messages save in inbox");
        sents.add("very good experienc...");
        sents.add("easy to retrieve mail, wants to logg in.");
        sents.add("a couple of friends use outook now and are very happy with it.");
        sents.add("shuld be more modern, interactive and losts of cool options");
        sents.add("there was a time i couldnt fault this app but lately its not been good,");
        sents.add("didn't want to show my latest email that had been sent which is one of rhe most important features.");
        sents.add("i have uninstalled and im going to reinstall hoping this will do the trick.");
        sents.add("says email address and password not valid which type thye are.");
        sents.add("thank you so muvh");
        sents.add("one has to scroll all the way down in inbox to tap on \\\"load more emails\\\" repeatedly to download a few emails at a timr.");
        sents.add("very good aplication");

        List<String> corrections = new ArrayList<>();
        List<Integer> tokenLengths = new ArrayList<>();

        for (String text : sents) {
            List<String> tokens = tokenizer.matchWordsAndSymbols(text.toLowerCase());

            List<String> correction = nGramModel.spell(tokens, dict_deletes, false, 0.90, 100_000, List.of("to", "too", "ad", "add", "ads", "adds")); // be less conservative

            tokenLengths.add(tokens.size());

            // Combine correctly
            StringBuilder builder = new StringBuilder();
            for (String token : correction) {
                if (!NLPUtilities.hasLatinAlphabet(token)) {
                    builder.append(token);
                } else {
                    builder.append(" "+token);
                }
            }

            System.out.println("Original: " + text);
            System.out.println(builder.toString().strip());

            corrections.add(builder.toString().strip());
        }

        String filePath = "H:\\Laptop\\last_3yeras_50k\\correction_results.txt";

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {

            for (int i = 0; i < corrections.size(); i++) {
                writer.write("OG: " + sents.get(i));
                writer.newLine();
                writer.write("CO: " + corrections.get(i));
                writer.newLine();
                writer.write("TokenLengths: " + tokenLengths.get(i));
                writer.newLine();
                writer.newLine();
                
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        // int counts = 0;

        // String line = "rhe best";

        // for (int i = 0; i < 1; i++) {
        //     counts++;

        //     if (counts % 100 == 0) {
        //         System.out.println("Processed: " + counts);
        //     }

        //     List<String> tokens = tokenizer.matchWordsAndSymbols(line.toLowerCase());

        //     List<String> correction = nGramModel.spell(tokens, dict_deletes, false, 0.01, 100_000, List.of("ads", "ad", "add", "adds", "to", "too")); // be less conservative

        //     System.out.println("Original: " + line);
        //     System.out.println(Arrays.toString(correction.toArray(new String[0])));
        // }

        // String app = "whatsapp";

        // String filePath = "h:\\laptop\\last_3yeras_50k\\"+app+"_results_splitted.txt";  // Replace with the path to your file
        // String outputFilePath = "h:\\laptop\\last_3yeras_50k\\"+app+"_results_splitted_corrected.txt";

        // boolean verbose = false;

        // try (BufferedReader reader = new BufferedReader(new FileReader(filePath));
        //     BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {

        //     String line;
        //     int counts = 0;

        //     while ((line = reader.readLine()) != null) {
        //         counts++;

        //         if (counts % 100 == 0) {
        //             System.out.println("Processed: " + counts);
        //         }

        //         List<String> tokens = tokenizer.matchWordsAndSymbols(line.toLowerCase());
        //         List<String> correction = nGramModel.spell(tokens, dict_deletes, false, 0.99, 100_000, List.of("to", "too", "ad", "add", "ads", "adds")); // be less conservative

        //         // Write the original and corrected line to the file
        //         // writer.write("Original: " + line + "\n");
                
        //         // Combine correctly
        //         StringBuilder builder = new StringBuilder();
        //         for (String token : correction) {
        //             if (!NLPUtilities.hasLatinAlphabet(token)) {
        //                 builder.append(token);
        //             } else {
        //                 builder.append(" "+token);
        //             }
        //         }
        //         String newSent = builder.toString().strip();

        //         writer.write(newSent + "\n");
        //         // writer.write("\n"); // Add a newline between entries

        //     }

        // } catch (IOException e) {
        //     e.printStackTrace();
        // }

        // String filePath = "H:\\Laptop\\last_3yeras_50k\\correction_results_evaluated.txt";
        // List<String> eval = new ArrayList<>();
        // List<Integer> tkLength = new ArrayList<>();

        // try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {

        //     String line;
        //     int counts = 0;

        //     while ((line = reader.readLine()) != null) {
        //         if (counts == 3 && eval.isEmpty()) {
        //             eval.add(line.strip());
        //             counts = 0;
        //         } else if (line.contains("TokenLengths:")) {
        //             tkLength.add(Integer.parseInt(line.strip().split(":")[1].strip()));
        //         } else if ((counts % 5 == 0) && (counts > 3)) {
        //             eval.add(line.strip());
        //         }
        //         counts++;
        //     }

        // } catch (IOException e) {
        //     e.printStackTrace();
        // }

        // List<Integer> tp = new ArrayList<>();
        // List<Integer> tn = new ArrayList<>();
        // List<Integer> fp = new ArrayList<>();
        // List<Integer> fn = new ArrayList<>();

        // for (int i = 0; i < eval.size(); i++) {
        //     String[] metrics = eval.get(i).split(",");

        //     int truePositive = Integer.parseInt(metrics[0].split(":")[1].strip());
        //     tp.add(truePositive);
        //     int falsePositive = Integer.parseInt(metrics[2].split(":")[1].strip());
        //     fp.add(falsePositive);
        //     int falseNegative = Integer.parseInt(metrics[3].split(":")[1].strip());
        //     fn.add(falseNegative);
        //     tn.add(Integer.parseInt(metrics[1].split(":")[1].strip()) + tkLength.get(i) - truePositive - falsePositive - falseNegative);

        //     // for (String m : metrics) {
        //     //     String metric = m.split(":")[1].strip();
        //     //     System.out.println(metric);
        //     // }
        // }

        // System.out.println("All true positives");
        // System.out.println(tp.stream().mapToInt(Integer::intValue).sum());
        // int tpValue = tp.stream().mapToInt(Integer::intValue).sum();
        // System.out.println("All false positives");
        // System.out.println(fp.stream().mapToInt(Integer::intValue).sum());
        // int fpValue = fp.stream().mapToInt(Integer::intValue).sum();
        // System.out.println("All false negatives");
        // System.out.println(fn.stream().mapToInt(Integer::intValue).sum());
        // int fnValue = fn.stream().mapToInt(Integer::intValue).sum();
        // System.out.println("All true negatives");
        // System.out.println(tn.stream().mapToInt(Integer::intValue).sum());
        // int tnValue = tn.stream().mapToInt(Integer::intValue).sum();

        // System.out.println("Precision: " + ((double) tpValue/(tpValue+fpValue)));
        // double precision = ((double) tpValue/(tpValue+fpValue));
        // System.out.println("Recall: " + ((double) tpValue/(tpValue+fnValue)));
        // double recall = ((double) tpValue/(tpValue+fnValue));
        // System.out.println("f1-score: " + ((double) 2*(precision*recall)/(precision+recall)));
        // double f1Score = (double) 2*(precision*recall)/(precision+recall);

        // System.out.println("Accuracy: " + (double) (tpValue+tnValue)/(tpValue+tnValue+fpValue+fnValue));
        
        
    }
}