# Limitations, in plain words

- **Side view, rigid objects, still camera.** Foresight models balls and boxes moving in the image plane. A ball
  rolling toward the camera, a cloth or a liquid is out of scope.
- **Short motion.** In Physics-IQ the object often moves for only the last 0.2 to 0.6 s of the 3 s it is shown,
  so the fit leans on gravity and the object's known size.
- **Scale comes from Gemini's size estimate.** A wrong size (a 6.7 cm tennis ball called 10 cm) scales every
  distance by the same factor; falls are cross-checked against gravity, rolls are not.
- **A subset of Physics-IQ.** 15 of 66 scenes and one of three cameras, so the scores are not comparable with the
  public leaderboard.
- **Physics is not yet better than a straight line on this score.** Contacts and falls are where the fit breaks;
  the README's results name the clips. Constant velocity shares Foresight's perception and tracking, so most of the
  gap to Omni comes from reading the real scene rather than generating pixels.
- **The score likes stillness.** Holding the last frame scores well on Physics-IQ because the pixel-error term
  rewards a video that does not change. The table shows it so you can judge.
- **Omni is slow and paid.** It took a median 38.9 s a clip in the eval and is billed per second of video, so
  the stage asks before calling it.
