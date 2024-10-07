package org.NGramModel;

import java.util.Comparator;
import java.util.PriorityQueue;

public class TopSuggestions {
    private PriorityQueue<Suggestion> suggestionsQueue;

    // Constructor
    public TopSuggestions() {
        // PriorityQueue with custom comparator to sort by score in ascending order.
        // We want a max heap, so reverse the natural order.
        suggestionsQueue = new PriorityQueue<>(Comparator.comparingDouble(Suggestion::getScore));
    }

    public void clear() {
        suggestionsQueue.clear();
    }

    // Method to add a suggestion
    public void addSuggestion(Suggestion suggestion) {
        // Add the suggestion to the queue
        if (suggestionsQueue.size() < 10) {
            suggestionsQueue.add(suggestion);
        } else if (suggestion.getScore() > suggestionsQueue.peek().getScore()) {
            // If the new suggestion is better than the worst one in the queue, replace it
            suggestionsQueue.poll(); // Remove the lowest score
            suggestionsQueue.add(suggestion);
        }
    }

    // Method to get top 10 suggestions in descending order
    public Suggestion[] getTopSuggestions() {
        return suggestionsQueue.stream()
                .sorted(Comparator.comparingDouble(Suggestion::getScore).reversed())
                .toArray(Suggestion[]::new);
    }
}
