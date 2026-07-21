// Package auth defines the API's verified request identity boundary.
package auth

import "context"

type principalContextKey struct{}

// Principal is an identity established by trusted authentication middleware.
// Request headers must never be copied into this type without verification.
type Principal struct {
	Subject string
	Roles   []string
}

// WithPrincipal attaches a verified principal to a request context.
func WithPrincipal(ctx context.Context, principal Principal) context.Context {
	roles := append([]string(nil), principal.Roles...)
	principal.Roles = roles
	return context.WithValue(ctx, principalContextKey{}, principal)
}

// HasRole reports whether a verified, identified principal has an exact role.
func HasRole(ctx context.Context, requiredRole string) bool {
	if requiredRole == "" {
		return false
	}
	principal, ok := ctx.Value(principalContextKey{}).(Principal)
	if !ok || principal.Subject == "" {
		return false
	}
	for _, role := range principal.Roles {
		if role == requiredRole {
			return true
		}
	}
	return false
}
