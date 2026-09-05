# Small real Razorpay Test Mode flow

This MVP intentionally implements only the useful read-only slice.

## 1. Get Test Mode credentials

In the Razorpay Dashboard, switch to **Test Mode** and generate a Test Mode Key ID and Key Secret. Test Mode is a sandbox; no real money moves.

## 2. Create a merchant session

Register once:

```bash
curl -X POST http://localhost:5000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"merchant@example.com","password":"demo1234","organization_id":"shop_001"}'
```

Login and keep the returned token:

```bash
curl -X POST http://localhost:5000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"merchant@example.com","password":"demo1234"}'
```

## 3. Connect Test Mode

Use the Test Mode credentials. The server makes a real read-only request to Razorpay. It never performs a payment/refund action.

```bash
curl -X POST http://localhost:5000/integrations/razorpay/test-connection \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"key_id":"rzp_test_...","key_secret":"...","webhook_secret":"...","data_source_id":"rzp_shop_001"}'
```

A successful response confirms the Test Mode connection and returns the webhook endpoint. Secrets are never returned.

## 4. Import payment/refund history

```bash
curl -X POST http://localhost:5000/integrations/razorpay/import \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"data_source_id":"rzp_shop_001"}'
```

The imported canonical data contains payment/order value and `refund_event` where a refund exists. It deliberately does **not** invent `return_event` from a refund.

## 5. Add one Test Mode webhook

In Razorpay Dashboard → Test Mode → Webhooks, configure:

```text
https://<your-public-host>/webhooks/razorpay/rzp_shop_001
```

Use the same webhook secret supplied above and enable one useful payment/refund event for the demo.

The receiver:

- verifies `X-Razorpay-Signature` against the raw request body using HMAC-SHA256
- uses the configured data source to resolve the merchant organization
- deduplicates `x-razorpay-event-id`
- stores the event as a canonical operational event
- does not trigger refunds, blocking, cancellation, or other financial actions

Razorpay requires a public webhook URL; localhost cannot receive direct webhook delivery. For a local demo, use a suitable public staging/tunnel endpoint that Razorpay accepts.

## Product story

**Razorpay tells us what happened to the payment. The merchant's returns/OMS data tells us whether the order was actually returned.**

So the flow stays simple:

```text
Razorpay Test Mode
      ↓
payments + refunds
      ↓
merchant canonical data
      ↓
return-risk model (only when true return labels exist)
```
