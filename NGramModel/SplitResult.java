package org.NGramModel;

import java.util.List;

public class SplitResult {
    protected List<String> words;
    protected long totalFrequency;

    public SplitResult(List<String> words, long totalFrequency) {
        this.words = words;
        this.totalFrequency = totalFrequency;
    }

    public List<String> getWords() {
        return words;
    }

    @Override
    public String toString() {
        return "SplitResult{" + "words=" + words + ", totalFrequency=" + totalFrequency + "}";
    }
}