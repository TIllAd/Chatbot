package de.fau.wisochatbot.chat;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.util.List;
import java.util.Map;
@Service
public class LlmService {

  private final WebClient client;
  private final String model;

  public LlmService(
      WebClient.Builder builder,
      @Value("${llm.base-url}") String baseUrl,
      @Value("${llm.model}") String model) {
    this.client = builder.baseUrl(baseUrl).build();
    this.model = model;
  }

  public Map<?, ?> reply(String message, boolean debug) {
    try {
      Map<String, Object> body = Map.of("message", message);

      String uri = debug ? "/chat?debug=true" : "/chat";

      Map<?, ?> res =
          client
              .post()
              .uri(uri)
              .contentType(MediaType.APPLICATION_JSON)
              .bodyValue(body)
              .retrieve()
              .bodyToMono(Map.class)
              .block();

      if (res == null) return Map.of("reply", "(keine Antwort)");
      return res;

    } catch (WebClientResponseException e) {
      return Map.of("reply", "⚠️ Fehler: HTTP " + e.getStatusCode().value());
    } catch (Exception e) {
      return Map.of("reply", "⚠️ Unerwarteter Fehler: " + e.getMessage());
    }
  }
}